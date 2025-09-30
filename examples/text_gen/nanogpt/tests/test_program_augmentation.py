from pathlib import Path
import importlib.util

import numpy as np
import torch

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

    batch = pipeline.get_batch_with_program("train")

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
