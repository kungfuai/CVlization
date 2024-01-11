import torch


class ScoreNetworkWithEnergy(torch.nn.Module):
    def __init__(self, net, prediction_type: str = "epsilon", **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.prediction_type = prediction_type

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
        energy = energy.mean(dim=dims)
        return energy  # (batch_size,)

    def __call__(self, x, timesteps, **kwargs):
        # return the gradient of energy on x
        e = self.energy(x, timesteps, **kwargs)
        grad = torch.autograd.grad(e.sum(), x, create_graph=True)[0]
        return grad
