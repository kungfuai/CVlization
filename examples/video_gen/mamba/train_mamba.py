import json
import argparse

import asciichartpy
from termcolor import colored
import numpy as np
import torch
from tqdm import tqdm

from mamba_classifier import MambaClassifier

def load_data() -> torch.Tensor:
    data = np.load("flying_mnist_tokens_32frames_train.npy")
    return torch.tensor(data, dtype=torch.long)

def calc_loss_for_one_batch(
    model: MambaClassifier,
    sequence: torch.Tensor,
) -> torch.Tensor:
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
    # sequence -> (B, L).
    # Remove last token from sequence (last token is a target only).
    logits = model(sequence) # (B, L, n_tokens)
    targets = sequence # (B, L)
    assert logits.shape[:-1] == targets.shape, f"{logits.shape[:-1]} != {targets.shape}"

    # torch cross entropy loss expects (B, C, L) and (B, L) shapes.
    logits = logits.permute(0, 2, 1) # (B, C, L)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, targets)

    return loss

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
    losses = []
    for i in pbar:
        batch = data[i:i+batch_size]
        optimizer.zero_grad()
        loss = calc_loss_for_one_batch(model, batch)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        losses.append(loss.item())
    print(f"Epoch {epoch} average loss: {np.mean(losses)}")
    print(colored(f"End epoch {epoch} TRAINING", "red", "on_grey", attrs=["bold"]))
    print()

    return np.mean(losses)

def run_val_epoch(
    model: MambaClassifier,
    data: torch.Tensor,
    batch_size: int,
    epoch: int,
) -> float:
    print(f"Validating epoch {epoch}")
    model.eval()
    pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}")
    losses = []
    with torch.no_grad():
        for i in pbar:
            batch = data[i:i+batch_size]
            loss = calc_loss_for_one_batch(model, batch)
            pbar.set_postfix({"loss": loss.item()})
            losses.append(loss.item())
    print(f"Epoch {epoch} average loss: {np.mean(losses)}")
    print(colored(f"End epoch {epoch} VALIDATION", "green", "on_grey", attrs=["bold"]))
    print()

    return np.mean(losses)

def plot_epochs(train_losses, val_losses, epoch: str):
    print(f"Epoch {epoch} losses (train = red, val = green)")
    config = {
        "colors": [asciichartpy.red, asciichartpy.green],
        "height": 10,
    }
    print(asciichartpy.plot([train_losses, val_losses], config))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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

    data: torch.Tensor = load_data().to(args.device)
    model = MambaClassifier(
        n_tokens=5120,
        # seq_len=32768,
        seq_len=data.shape[1],
        mamba_n_embed=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        n_mamba_layers=1,
        device=args.device,
    ).to(args.device)

    train_size = int(len(data) * args.train_frac)
    train_data = data[:train_size]
    val_data = data[train_size:]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # run_train_epoch(model, train_data, optimizer, batch_size, 0)
    train_epoch_losses = []
    val_epoch_losses = []
    def check_save_model():
        return val_epoch_losses[-1] == min(val_epoch_losses)
    def check_patience_quit():
        if len(val_epoch_losses) <= args.patience:
            return False # Not enough samples.
        best_val_loss = min(val_epoch_losses) # Best metric.
        if min(val_epoch_losses[-args.patience:]) > best_val_loss:
            return True # The last <patience> epochs were worse, so quit.
        return False
    for epoch in range(args.epochs):
        epoch_loss = run_train_epoch(model, train_data, optimizer, args.batch_size, epoch)
        train_epoch_losses.append(epoch_loss)
        epoch_loss = run_val_epoch(model, val_data, args.batch_size, epoch)
        val_epoch_losses.append(epoch_loss)
        if epoch > args.start_plotting_after_epoch:
            # Remove first epoch cause it messes up the graph.
            train_to_plot = train_epoch_losses[args.start_plotting_after_epoch:]
            val_to_plot = val_epoch_losses[args.start_plotting_after_epoch:]
            plot_epochs(train_to_plot, val_to_plot, str(epoch))
        else:
            print(f"(Don't plot until epoch {args.start_plotting_after_epoch})")
        if check_save_model():
            print(colored(f"Saving model at epoch {epoch}", "green", "on_grey", attrs=["bold"]))
            torch.save(model.state_dict(), "mamba_model.pth")
        elif check_patience_quit():
            print(colored(f"Early stopping at epoch {epoch}", "yellow", "on_grey", attrs=["bold"]))
            break

    print(colored("EXPERIMENT COMPLETE.", "blue", "on_grey", attrs=["bold"]))
    print("\nBye-bye.")

