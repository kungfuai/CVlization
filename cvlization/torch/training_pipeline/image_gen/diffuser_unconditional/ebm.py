import torch


class ScoreNetworkOutput:
    def __init__(self, sample):
        self.sample = sample


class ScoreNetworkWithEnergy(torch.nn.Module):
    def __init__(self, net, prediction_type: str = "epsilon", **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.prediction_type = prediction_type
        self.device = net.device
        self.sample_size = net.sample_size
        self.in_channels = net.in_channels
        self.out_channels = net.out_channels

    def energy(self, x, timesteps, **kwargs):
        model_output = self.net(
            x, timesteps, **kwargs
        ).sample  # .detach()  # stop gradient?
        if self.prediction_type == "epsilon":
            energy = model_output**2
        elif self.prediction_type == "sample":
            energy = (model_output - x) ** 2
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        dims = list(range(1, energy.ndim))
        energy = energy.sum(dim=dims)
        return energy  # (batch_size,)

    def __call__(self, x, timesteps, create_graph=False, **kwargs):
        # return the gradient of energy on x
        # turn on grad
        x_requires_grad = x.requires_grad
        x.requires_grad_(True)
        torch.set_grad_enabled(True)
        e = self.energy(x, timesteps, **kwargs)
        grad = torch.autograd.grad(e.sum(), [x], create_graph=create_graph)[0]
        # grad = torch.autograd.grad(e.sum(), [x], create_graph=True)[0]
        # turn off grad
        torch.set_grad_enabled(x_requires_grad)
        x.requires_grad_(x_requires_grad)
        output = ScoreNetworkOutput(sample=grad)
        return output
