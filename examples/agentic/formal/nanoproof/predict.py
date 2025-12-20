import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

# Make vendored nanoproof importable
VENDOR_ROOT = Path(__file__).resolve().parent / "nanoproof_repo"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))

from nanoproof.common import get_base_dir, print0  # noqa: E402


class _FakeBranch:
    """Minimal shim to satisfy TacticModel.sample_tactic (expects .state)."""

    def __init__(self, state_text: str):
        self.state = state_text


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_tokenizer() -> Path:
    base_dir = Path(get_base_dir())
    target = base_dir / "tokenizer"
    tokenizer_json = target / "tokenizer.json"
    token_bytes = target / "token_bytes.pt"
    if tokenizer_json.exists() and token_bytes.exists():
        return target

    local_override = os.environ.get("NANOPROOF_TOKENIZER_LOCAL")
    if local_override:
        src = Path(local_override)
        if not (src / "tokenizer.json").exists() or not (src / "token_bytes.pt").exists():
            raise FileNotFoundError(f"tokenizer.json/token_bytes.pt not found in {src}")
        ensure_dir(target)
        (target / "tokenizer.json").write_bytes((src / "tokenizer.json").read_bytes())
        (target / "token_bytes.pt").write_bytes((src / "token_bytes.pt").read_bytes())
        print0(f"Copied tokenizer from {src}")
        return target

    repo_id = os.environ.get("NANOPROOF_TOKENIZER_REPO_ID")
    revision = os.environ.get("NANOPROOF_TOKENIZER_REVISION")
    if not repo_id:
        raise RuntimeError(
            "Tokenizer missing. Set NANOPROOF_TOKENIZER_REPO_ID (HF repo with tokenizer.json/token_bytes.pt) "
            "or mount a local path via NANOPROOF_TOKENIZER_LOCAL."
        )

    print0(f"Downloading tokenizer from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["tokenizer.json", "token_bytes.pt"],
        local_dir=target,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    return target


def ensure_checkpoint(model_tag: str) -> Path:
    base_dir = Path(get_base_dir())
    target = base_dir / "sft_checkpoints" / model_tag
    model_glob = list(target.glob("model_*.pt"))
    meta_glob = list(target.glob("meta_*.json"))
    if model_glob and meta_glob:
        return target

    local_override = os.environ.get("NANOPROOF_CHECKPOINT_LOCAL")
    if local_override:
        src = Path(local_override)
        ensure_dir(target)
        for file in src.glob("*"):
            if file.is_file():
                (target / file.name).write_bytes(file.read_bytes())
        print0(f"Copied checkpoint from {src}")
        return target

    repo_id = os.environ.get("NANOPROOF_CHECKPOINT_REPO_ID")
    revision = os.environ.get("NANOPROOF_CHECKPOINT_REVISION")
    if not repo_id:
        raise RuntimeError(
            "Checkpoint missing. Set NANOPROOF_CHECKPOINT_REPO_ID (HF repo with sft_checkpoints/<model_tag>/...) "
            "or mount a local path via NANOPROOF_CHECKPOINT_LOCAL."
        )

    print0(f"Downloading checkpoint {model_tag} from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[
            f"sft_checkpoints/{model_tag}/model_*.pt",
            f"sft_checkpoints/{model_tag}/meta_*.json",
        ],
        local_dir=base_dir,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample tactics using nanoproof.")
    parser.add_argument("--state", type=str, required=True, help="Lean tactic state (string).")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of tactics to sample.")
    parser.add_argument("--model-tag", type=str, default=os.environ.get("NANOPROOF_MODEL_TAG", "d26"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_dir = ensure_tokenizer()
    ckpt_dir = ensure_checkpoint(args.model_tag)
    print0(f"Tokenizer ready at {tokenizer_dir}")
    print0(f"Checkpoint ready at {ckpt_dir}")

    # Defer heavy imports until assets exist
    from nanoproof.search import TacticModel  # noqa: E402

    model = TacticModel.create()
    tactics = model.sample_tactic([_FakeBranch(args.state)], num_samples=args.num_samples)
    print0("=== Sampled tactics ===")
    for i, tactic in enumerate(tactics, 1):
        print0(f"[{i}] {tactic}")


if __name__ == "__main__":
    main()
