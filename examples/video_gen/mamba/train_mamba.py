import json
from typing import Optional, Union
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
from examples.video_gen.minisora.nanogpt import GPT, GPTConfig
from cvlization.torch.net.vae.video_vqvae import VQVAE

# seed numpy's random
np.random.seed(0)

IGNORE_TOKEN = 5120

hyperparams = {
    "num_tokens_to_mask": 100,
    "num_seq_gen_steps": 100,
    "gen_prob_thresh": 0.01,
    "num_data_samples": 1000,
}

def load_data() -> torch.Tensor:
    # data = np.load("flying_mnist_tokens_32frames_train.npy")
    data = np.load("flying_mnist_11k_tokens_32frames_train.npy")
    assert data.shape == (10000, 8, 64, 64), f"Expected (1000, 8, 64, 64), got {data.shape}"
    assert data.max() < IGNORE_TOKEN, f"Expected max token to be ({IGNORE_TOKEN} - 1), got {data.max()}"
    # Move `8` to last dim, then flatten after batch dimension.
    # data = np.moveaxis(data, 1, -1)
    # assert data.shape == (1000, 64, 64, 8), f"Expected (1000, 64, 64, 8), got {data.shape}"
    # data = data.reshape(-1, 8*64*64)
    # assert data.shape == (1000, 8*64*64), f"Expected (1000, 8*64*64), got {data.shape}"
    n_samples = hyperparams["num_data_samples"]
    return torch.tensor(data[:n_samples], dtype=torch.long)

def train_one_batch(
        model: Union[MambaClassifier, GPT],
        sequence: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer],
        device: str,
) -> float:
    """
    (b, video_dim) -> (b, 8, 64, 64)

    Classifier model is not a standard classifier - it's going to return
    a tensor of shape (B, L, C) where B is the batch size, L is the sequence length,
    and C is the classification for each item in the sequence.

    Mask random tokens and predict.
    """
    assert sequence.shape[1:] == (8, 64, 64), f"Expected (b, 8, 64, 64), got {sequence.shape}"
    num_tokens_to_mask = hyperparams["num_tokens_to_mask"]

    subsequence = sequence[:, 0] # First clip.
    assert subsequence.shape == (sequence.shape[0], 64, 64), f"Expected (b, 64, 64), got {subsequence.shape}"

    # generate random tokens the same shape as batch sequence.
    targets = subsequence.view(subsequence.shape[0], int(64*64))
    model_input = targets.clone()
    mask_indices = torch.randint(0, 64*64, (num_tokens_to_mask,))
    model_input[:, mask_indices] = IGNORE_TOKEN

    if isinstance(model, GPT):
        logits, loss = model(
            model_input.to(device),
            targets=targets.to(device),
        )
    else:
        logits = model(model_input.to(device))
        assert logits.shape == (sequence.shape[0], 64*64, IGNORE_TOKEN + 1), f"Expected (b, 64*64, IGNORE_TOKEN + 1), got {logits.shape}"
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        # torch cross entropy loss expects (B, C, L) and (B, L) shapes.
        loss = criterion(logits.permute(0, 2, 1), targets.to(device))

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def run_train_epoch(
    model: Union[GPT, MambaClassifier],
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epoch: int,
    device: str,
) -> float:
    print(f"Training epoch {epoch}")
    model.train()
    pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}")
    total_loss = 0
    count = 0
    for i in pbar:
        batch = data[i:i+batch_size]
        batch_loss: float = train_one_batch(model, batch, optimizer=optimizer, device=device)
        pbar.set_postfix({"train_batch_loss": batch_loss})
        total_loss += batch_loss
        count += 1
    avg_loss = total_loss / count

    print(f"Epoch {epoch} average loss: {avg_loss}")
    print(colored(f"End epoch {epoch} TRAINING", "red", "on_grey", attrs=["bold"]))
    print()

    return avg_loss

def run_val_epoch(
    model: Union[GPT, MambaClassifier],
    data: torch.Tensor,
    batch_size: int,
    epoch: int,
    device: str,
) -> float:
    print(f"Validating epoch {epoch}")
    model.eval()
    pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}")
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in pbar:
            batch = data[i:i+batch_size]
            batch_loss: float = train_one_batch(model, batch, optimizer=None, device=device)
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

def generate_sequence(mamba: torch.nn.Module, device: str, init_with_random_tokens: bool = False) -> torch.Tensor:
    """
    Either initialize with random tokens or use the IGNORE_TOKEN.

    1. Predict tokens.
    2. Replace all tokens of softmax-proba > 0.5 with the predicted token.
    Repeat num_steps times, or until no tokens are replaced. (early stopping)
    """
    num_steps = hyperparams["num_seq_gen_steps"]

    if init_with_random_tokens:
        print("Initializing with random tokens.")
        values = torch.randint(0, IGNORE_TOKEN, (1, 64*64), device=device)
    else:
        print("Initializing with IGNORE_TOKEN.")
        values = torch.full((1, 64*64), IGNORE_TOKEN, device=device)
    assert values.shape == (1, 64*64), f"Expected (1, 64*64), got {values.shape}"

    predictions = None
    proba_thresh = hyperparams["gen_prob_thresh"]
    with torch.no_grad():
        for step in tqdm(range(num_steps), desc="Generating sequence"):
            logits = mamba(values)[0]
            # run softmax on all predictions.
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            # replace all tokens with max_probs > proba_thresh with the predicted token.
            diff_tokens = torch.sum(
                (values != predictions) & (max_probs > proba_thresh),
            )
            if diff_tokens == 0:
                print(f"[Step {step}: Early stopping. No tokens changed.")
                break
            values[max_probs > proba_thresh] = predictions[max_probs > proba_thresh]
            print(f"[Step {step}: Replaced {torch.sum(max_probs > proba_thresh)} tokens, and {diff_tokens} of them changed.")

    # replace any IGNORE_TOKEN with prediction.
    assert predictions is not None
    values[values == IGNORE_TOKEN] = predictions[values == IGNORE_TOKEN]

    assert values.shape == (1, 64*64), f"Expected (1, 64*64), got {values.shape}"
    # repeat 8 times on new axis.
    values = values.view(1, 64, 64).repeat(8, 1, 1).unsqueeze(0)
    assert values.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {values.shape}"
    return values

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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start_plotting_after_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    # print args as json. print in bold green.
    print("Arguments:")
    print(colored(json.dumps(vars(args), indent=4), "cyan"), "\n\n")

    wandb.init(project="mamba_videos_it")

    mamba_kwargs = {
        "mamba_n_embed": 512,
        "mamba_d_state": 32,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "n_mamba_layers": 4,
    }

    # Log args and mamaba_kwargs to wandb.
    wandb.config.update(vars(args))
    wandb.config.update(mamba_kwargs)
    wandb.config.update(hyperparams)

    data: torch.Tensor = load_data()
    # model = MambaClassifier(
    #     n_tokens=IGNORE_TOKEN + 1, # 5120 tokens + 1 IGNORE.
    #     seq_len=32768,
    #     # seq_len=int(2*64*64), # Condition on last 2 frames to predict the next frame.
    #     **mamba_kwargs,
    #     device=args.device,
    # ).to(args.device)
    model = GPT(
        GPTConfig(
            block_size=4096,
            vocab_size=IGNORE_TOKEN + 1,
            dropout=0.1,
        ),
    ).to(args.device)

    vae_device = "cuda:1" # args.device
    vae = load_vae().to(vae_device)

    train_size = int(len(data) * args.train_frac)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_vid = data[0].unsqueeze(0)
    # val_vid = val_data[0].unsqueeze(0)

    train_sanity_vid = decode_video(vae, train_vid.to(vae_device))
    # val_sanity_vid = decode_video(vae, val_vid.to(vae_device))

    wandb.log({
        "train_sanity_vid": wandb.Video(train_sanity_vid, fps=5, format="mp4"),
        # "val_sanity_vid": wandb.Video(val_sanity_vid, fps=5, format="mp4"),
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
        epoch_loss = run_train_epoch(model, train_data, optimizer, args.batch_size, epoch=epoch, device=args.device)
        # log train_epoch_loss metric to wandb.
        to_log = {"train_epoch_loss": epoch_loss}
        train_epoch_losses.append(epoch_loss)
        epoch_loss = run_val_epoch(model, val_data, args.batch_size, epoch=epoch, device=args.device)
        to_log["val_epoch_loss"] = epoch_loss
        if epoch % 2 == 0:
            # Generate a video.
            with torch.no_grad():
                ignore_generated_sequence = generate_sequence(model, args.device, init_with_random_tokens=False)
                random_generated_sequence = generate_sequence(model, args.device, init_with_random_tokens=True)
                ignore_generated_vid = decode_video(vae, ignore_generated_sequence.to(vae_device))
                random_generated_vid = decode_video(vae, random_generated_sequence.to(vae_device))
                to_log["ignore_generated_vid"] = wandb.Video(ignore_generated_vid, fps=5, format="mp4")
                to_log["random_generated_vid"] = wandb.Video(random_generated_vid, fps=5, format="mp4")
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

