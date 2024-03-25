import json
from typing import Optional
import argparse
import os

from einops import rearrange
from termcolor import colored
from tqdm import tqdm
import asciichartpy
import numpy as np
import torch
import wandb

from examples.video_gen.mamba.mamba_classifier import MambaClassifier
from cvlization.torch.net.vae.video_vqvae import VQVAE

# seed numpy's random
np.random.seed(0)

def load_data() -> torch.Tensor:
    data = np.load("flying_mnist_tokens_32frames_train.npy")
    assert data.shape == (1000, 8, 64, 64), f"Expected (1000, 8, 64, 64), got {data.shape}"
    # Move `8` to last dim, then flatten after batch dimension.
    # data = np.moveaxis(data, 1, -1)
    # assert data.shape == (1000, 64, 64, 8), f"Expected (1000, 64, 64, 8), got {data.shape}"
    # data = data.reshape(-1, 8*64*64)
    # assert data.shape == (1000, 8*64*64), f"Expected (1000, 8*64*64), got {data.shape}"
    return torch.tensor(data, dtype=torch.long)

def train_one_batch(
        model: MambaClassifier,
        sequence: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer],
) -> float:
    """
    (b, video_dim) -> (b, 8, 64, 64)
    """
    """
    Imagine this sequence:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Next token predictions are:
    [0] -> 1,
    [0, 1] -> 2,
    [0, 1, 2] -> 3,
    etc.

    Classifier model is not a standard classifier - it's going to return
    a tensor of shape (B, L, C) where B is the batch size, L is the sequence length,
    and C is the classification for each item in the sequence.

    Try this: average cross entropy for each token in the sequence all at once.
    (since Mamba is a recurrent model).
    """
    assert sequence.shape[1:] == (8, 64, 64), f"Expected (b, 8, 64, 64), got {sequence.shape}"
    # torch cross entropy loss expects (B, C, L) and (B, L) shapes.
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    total_loss = 0
    count = 0
    for ix in range(2, 8):
        if optimizer is not None:
            optimizer.zero_grad()

        """
        For 2, model(frame0&1) predicts frame 2.
        For 3, model(frame1&2) predicts frame 3,
        etc.
        For ix=7, model(frame5&6) predicts frame 7.
        """
        b = sequence.shape[0]
        # targets: current clip.
        targets = sequence[:, ix].view(b, -1) # (B, 64, 64) -> (B, 64*64)
        # model_input: previous 2 clips.
        model_input = sequence[:, ix-2:ix].view(b, -1) # (B, 2, 64, 64) -> (B, 2*64*64)

        logits = model(model_input)[:, -int(64*64):] # (B, 2*64*64, C) -> (B, 64*64, C)

        """
        Cross entropy expects logits.shape == (B, C, D1, D2, ...),
        targets.shape == (B, D1, D2...)
        """
        loss = criterion(
            logits.permute(0, 2, 1), # (B, 64*64, C) -> (B, C, 64*64)
            targets,
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count

def run_train_epoch(
    model: MambaClassifier,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epoch: int,
) -> float:
    print(f"Training epoch {epoch}")
    model.train()
    pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}")
    total_loss = 0
    count = 0
    for i in pbar:
        batch = data[i:i+batch_size]
        batch_loss: float = train_one_batch(model, batch, optimizer=optimizer)
        pbar.set_postfix({"train_batch_loss": batch_loss})
        total_loss += batch_loss
        count += 1
    avg_loss = total_loss / count

    print(f"Epoch {epoch} average loss: {avg_loss}")
    print(colored(f"End epoch {epoch} TRAINING", "red", "on_grey", attrs=["bold"]))
    print()

    return avg_loss

def run_val_epoch(
    model: MambaClassifier,
    data: torch.Tensor,
    batch_size: int,
    epoch: int,
) -> float:
    print(f"Validating epoch {epoch}")
    model.eval()
    pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}")
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in pbar:
            batch = data[i:i+batch_size]
            batch_loss: float = train_one_batch(model, batch, optimizer=None)
            pbar.set_postfix({"val_batch_loss": batch_loss})
            total_loss += batch_loss
            count += 1
    avg_loss = total_loss / count

    print(f"Epoch {epoch} average loss: {avg_loss}")
    print(colored(f"End epoch {epoch} VALIDATION", "green", "on_grey", attrs=["bold"]))
    print()

    return avg_loss

def plot_epochs(train_losses, val_losses, epoch: str):
    print(f"Epoch {epoch} losses (train = red, val = green)")
    config = {
        "colors": [asciichartpy.red, asciichartpy.green],
        "height": 10,
    }
    print(asciichartpy.plot([train_losses, val_losses], config))
    print()

def generate_sequence(mamba: torch.nn.Module, prompt_sequence: torch.Tensor, device: str) -> torch.Tensor:
    """
    Returns a tensor of shape (1, seq_len).
    """
    assert prompt_sequence.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {prompt_sequence.shape}"
    output_video = prompt_sequence[:, :2].to(device) # Prompt with first two clips.
    with torch.no_grad():
        for ix in range(2, 8):
            assert ix == output_video.shape[1], f"Expected {ix}, got {output_video.shape[1]}"
            model_input = output_video[:, ix-2:ix].view(1, -1)
            assert model_input.shape == (1, 2*64*64), f"Expected (1, 2*64*64), got {model_input.shape}"
            logits = mamba(model_input)[:,-int(64*64):] # (1, 2*64*64, 5120) -> (1, 64*64, 5120)
            # take the argmax of the last dimension.
            next_clip = logits.argmax(dim=-1).view(1, 64, 64)
            assert next_clip.shape == (1, 64, 64), f"Expected (1, 64, 64), got {next_clip.shape}"
            output_video = torch.cat([output_video, next_clip.unsqueeze(0)], dim=1)
    assert output_video.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {output_video.shape}"
    return output_video.detach().cpu()

def load_vae() -> torch.nn.Module:
    api = wandb.Api()

    model_full_name = "zzsi_kungfu/videogpt/model-kbu39ped:v11"

    artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
    if os.path.exists(artifact_dir):
        print(f"Model already exists at {artifact_dir}")
    else:
        artifact_dir = api.artifact(model_full_name).download()

    vae = VQVAE.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
    )

    vae.eval()

    return vae

def decode_video(
        vae: torch.nn.Module,
        sequence: torch.Tensor,
) -> torch.Tensor:
    t, h, w = 8, 64, 64
    # sequence = rearrange(sequence, "(b h w t) -> b t h w", b=1, t=t, h=h, w=w)
    assert sequence.shape == (1, t, h, w), f"expected (1, {t}, {h}, {w}), got {sequence.shape}"
    with torch.no_grad():
        z = vae.vq.codes_to_vec(sequence)
        assert len(z.shape) == 5
        assert z.shape == (1, 4, t, h, w)
        video = vae.decoder(z)
        video = (video - video.min()) / (
            video.max() - video.min() + 1e-6
        )
        video = (video * 255).to(torch.uint8)
        video = rearrange(video, "b c t h w -> t c h (b w)")
        assert video.shape[1] == 3, f"shape of video is {video.shape}"
        return video.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start_plotting_after_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    # print args as json. print in bold green.
    print("Arguments:")
    print(colored(json.dumps(vars(args), indent=4), "cyan"), "\n\n")

    wandb.init(project="mamba_videos")

    mamba_kwargs = {
        "mamba_n_embed": 256,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 4,
        "n_mamba_layers": 2,
    }

    # Log args and mamaba_kwargs to wandb.
    wandb.config.update(vars(args))
    wandb.config.update(mamba_kwargs)

    data: torch.Tensor = load_data().to(args.device)
    model = MambaClassifier(
        n_tokens=5120,
        # seq_len=32768,
        seq_len=int(2*64*64), # Condition on last 2 frames to predict the next frame.
        **mamba_kwargs,
        device=args.device,
    ).to(args.device)

    vae_device = "cuda:1" # args.device
    vae = load_vae().to(vae_device)

    train_size = int(len(data) * args.train_frac)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_vid = data[0].unsqueeze(0)
    val_vid = val_data[0].unsqueeze(0)

    train_sanity_vid = decode_video(vae, train_vid.to(vae_device))
    val_sanity_vid = decode_video(vae, val_vid.to(vae_device))

    wandb.log({
        "train_sanity_vid": wandb.Video(train_sanity_vid, fps=5, format="mp4"),
        "val_sanity_vid": wandb.Video(val_sanity_vid, fps=5, format="mp4"),
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # run_train_epoch(model, train_data, optimizer, batch_size, 0)
    train_epoch_losses = []
    val_epoch_losses = []
    def check_save_model():
        return val_epoch_losses[-1] == min(val_epoch_losses)
    def check_patience_quit():
        if len(val_epoch_losses) <= args.patience:
            return False # Not enough samples.
        best_val_loss_idx = np.argmin(val_epoch_losses) # Best metric index.
        if len(val_epoch_losses) - best_val_loss_idx > args.patience:
            return True
        print(f"Epochs since best val loss: {len(val_epoch_losses) - best_val_loss_idx}")
        return False
    for epoch in range(args.epochs):
        epoch_loss = run_train_epoch(model, train_data, optimizer, args.batch_size, epoch)
        # log train_epoch_loss metric to wandb.
        to_log = {"train_epoch_loss": epoch_loss}
        train_epoch_losses.append(epoch_loss)
        epoch_loss = run_val_epoch(model, val_data, args.batch_size, epoch)
        to_log["val_epoch_loss"] = epoch_loss
        if epoch % 10 == 0:
            # Generate a video.
            with torch.no_grad():
                train_generated_sequence = generate_sequence(model, train_vid, args.device)
                val_generated_sequence = generate_sequence(model, val_vid, args.device)
                train_generated_vid = decode_video(vae, train_generated_sequence.to(vae_device))
                val_generated_vid = decode_video(vae, val_generated_sequence.to(vae_device))
                to_log["train_generated_vid"] = wandb.Video(train_generated_vid, fps=5, format="mp4")
                to_log["val_generated_vid"] = wandb.Video(val_generated_vid, fps=5, format="mp4")
                print("Generated videos")
        wandb.log(to_log)
        if epoch_loss == 0.0:
            print(colored(f"Val loss is 0.0 at epoch {epoch}. Quitting.", "yellow", "on_grey", attrs=["bold"]))
            break
        val_epoch_losses.append(epoch_loss)
        print(colored(f"Best val loss so far is {min(val_epoch_losses)}.", "yellow"))
        if epoch > args.start_plotting_after_epoch:
            # Remove first epoch cause it messes up the graph.
            train_to_plot = train_epoch_losses[args.start_plotting_after_epoch:]
            val_to_plot = val_epoch_losses[args.start_plotting_after_epoch:]
            plot_epochs(train_to_plot, val_to_plot, str(epoch))
        else:
            print(f"(Don't plot until epoch {args.start_plotting_after_epoch + 1})")
        if check_save_model():
            print(colored(f"Saving model at epoch {epoch}", "green", "on_grey", attrs=["bold"]))
            torch.save(model.state_dict(), "mamba_model.pth")
            # Save optimizer.
            torch.save(optimizer.state_dict(), "mamba_optimizer.pth")
        elif check_patience_quit():
            print(colored(f"Early stopping at epoch {epoch}", "yellow", "on_grey", attrs=["bold"]))
            break

    print(colored("EXPERIMENT COMPLETE.", "blue", "on_grey", attrs=["bold"]))
    print("\nBye-bye.")

