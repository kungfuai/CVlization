import torch
import random
import numpy as np


class ScoreNetworkOutput:
    def __init__(self, sample):
        self.sample = sample


class ScoreNetworkWithEnergy(torch.nn.Module):
    def __init__(
        self,
        net,
        prediction_type: str = "epsilon",
        as_energy_net: bool = False,
        **kwargs,
    ):
        """
        as_energy_net: if True, the network is used as an energy network or as the score network.
        """
        super().__init__(**kwargs)
        self.net = net
        self.prediction_type = prediction_type
        self.device = net.device
        self.sample_size = net.sample_size
        self.in_channels = net.in_channels
        self.out_channels = net.out_channels
        self.as_energy_net = as_energy_net

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
        if self.as_energy_net:
            return self.energy(x, timesteps, **kwargs)
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


class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len=8192, device="cuda"):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.device = device
        self.examples = [
            (torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)
        ]

    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(
            random.choices(self.examples, k=self.sample_size - n_new), dim=0
        )
        device = self.device
        assert str(device).startswith(
            "cuda"
        ), f"examples device is {device}, but should be cuda"
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = (
            list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.examples
        )
        self.examples = self.examples[: self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(
        model,
        inp_imgs,
        timesteps,
        steps=60,
        step_size=10,
        return_img_per_step=False,
        clamp_gradients=False,
        clamp_pixels=False,
    ):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            timesteps - Noise levels.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            if clamp_pixels:
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            # TODO: negation or not?
            # out_imgs = -model(inp_imgs, timesteps)
            out_imgs = model(inp_imgs, timesteps)
            out_imgs.sum().backward()
            if clamp_gradients:
                inp_imgs.grad.data.clamp_(
                    -0.03, 0.03
                )  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            if clamp_pixels:
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
