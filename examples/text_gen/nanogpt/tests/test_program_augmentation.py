from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

import types
import sys

# Provide lightweight stubs so the NanoGPT module can import without heavy deps.
if "einops" not in sys.modules:
    einops_stub = types.ModuleType("einops")
    einops_stub.rearrange = lambda x, *args, **kwargs: x
    sys.modules["einops"] = einops_stub

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")

    def _noop(*args, **kwargs):  # pragma: no cover -- not expected in this unit test
        class _Dummy:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Dummy()

    wandb_stub.init = _noop
    wandb_stub.log = lambda *args, **kwargs: None
    wandb_stub.Video = object
    wandb_stub.Api = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub

_MODULE_PATH = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "cvlization" / "torch" / "training_pipeline" / "lm" / "gpt.py"
    if candidate.exists():
        _MODULE_PATH = candidate
        break

if _MODULE_PATH is None:
    raise FileNotFoundError("Could not locate NanoGPT pipeline module for testing")


spec = importlib.util.spec_from_file_location("_nanogpt_pipeline", _MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

NanoGPTTrainingPipeline = module.NanoGPTTrainingPipeline
ProgramAugmentedGPT = module.ProgramAugmentedGPT


class _DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 2
        self.config = types.SimpleNamespace(block_size=8, n_embd=hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, 3, bias=False)
        with torch.no_grad():
            self.lm_head.weight[:] = torch.tensor(
                [[-10.0, -10.0], [-10.0, -10.0], [10.0, 10.0]]
            )

    def forward(self, idx, targets=None):
        features = self.forward_features(idx)
        logits = self.lm_head(features)
        if targets is None:
            return logits[:, [-1], :], None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return logits, loss

    def forward_features(self, idx):
        b, t = idx.shape
        hidden = torch.ones(b, t, self.config.n_embd, device=idx.device)
        return hidden

    def generate(self, idx, *args, **kwargs):
        return idx

    def configure_optimizers(self, *args, **kwargs):
        return None

    def estimate_mfu(self, *args, **kwargs):
        return 0.0


class _DummyDatasetBuilder:
    def __init__(self, array: np.ndarray):
        self._array = array

    def training_dataset(self):
        return self._array

    def validation_dataset(self):
        return self._array


def test_get_batch_with_program_targets_are_masked_and_dense():
    program_offset = 100
    program_vocab_size = 5
    program_nil_id = program_offset + program_vocab_size
    data = np.array([1, program_offset, 2, program_offset + 3, 4], dtype=np.int32)

    config = NanoGPTTrainingPipeline.Config(
        batch_size=1,
        block_size=4,
        device="cpu",
        use_program_augmentation=True,
        program_offset=program_offset,
        program_nil_id=program_nil_id,
        program_vocab_size=program_vocab_size,
    )
    pipeline = NanoGPTTrainingPipeline(config)
    pipeline.create_dataloaders(_DummyDatasetBuilder(data))

    batch = pipeline.get_batch("train")

    assert batch["input_ids"].shape == (1, 4)
    torch.testing.assert_close(
        batch["targets"], torch.tensor([[program_offset, 2, program_offset + 3, 4]])
    )

    expected_text = torch.tensor([[-1, 2, -1, 4]])
    torch.testing.assert_close(batch["targets_text"], expected_text)

    nil_local = program_nil_id - program_offset
    expected_program = torch.tensor([[0, nil_local, 3, nil_local]])
    torch.testing.assert_close(batch["targets_program"], expected_program)

    assert pipeline.program_nil_local_id == nil_local


def test_program_augmented_generate_respects_gate():
    backbone = _DummyBackbone()
    pag = ProgramAugmentedGPT(
        backbone=backbone,
        program_vocab_size=2,
        program_offset=50,
        nil_local_id=2,
        nil_loss_weight=1.0,
    )

    # Force NIL path (use text head)
    with torch.no_grad():
        pag.program_head.weight.zero_()
        pag.program_head.weight[2].fill_(5.0)

    seed_idx = torch.tensor([[0]], dtype=torch.long)
    out_nil = pag.generate(seed_idx.clone(), max_new_tokens=1, top_k=1)
    assert out_nil[0, -1].item() == 2
