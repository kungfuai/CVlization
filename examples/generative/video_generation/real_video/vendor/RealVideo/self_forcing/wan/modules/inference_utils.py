import torch

COMPILE = False
torch._dynamo.config.cache_size_limit = 128

NO_REFRESH_INFERENCE = False

def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")

def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func

def conditional_compile(func):
    if COMPILE:
        return torch.compile(mode="default", backend="inductor", dynamic=True)(func)
    else:
        return func