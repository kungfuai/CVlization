import random
import torch
import numpy as np
from diffusers import DDPMScheduler
from .generator import SampleInitializer


class Sampler:
    def __init__(
        self,
        model,
        img_shape,
        sample_size,
        max_len=8192,
        diffusion_num_steps=100,
        use_sample_initializer=False,
        device="cuda",
    ):
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
        self.diffusion_num_steps = diffusion_num_steps
        self.device = device

        # This is a buffer for storing old samples.
        self.buffered_examples = [
            (torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)
        ]
        self.buffered_timesteps = [
            torch.randint(0, diffusion_num_steps, (1,)) for _ in range(self.sample_size)
        ]

        if use_sample_initializer:
            self.sample_initializer = SampleInitializer(
                latent_dim=32,
                img_shape=img_shape,
                diffusion_num_steps=diffusion_num_steps,
            )
        else:
            self.sample_initializer = None

    def sample_new_exmps(self, noisy_images, timesteps, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            noisy_images - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            timesteps - Timesteps in the diffusion schedule.
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        # TODO: noisy_images not used
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        rand_timesteps = torch.randint(0, self.diffusion_num_steps, (n_new,))
        old_imgs = torch.cat(
            random.choices(self.buffered_examples, k=self.sample_size - n_new), dim=0
        )
        old_timesteps = torch.cat(
            random.choices(self.buffered_timesteps, k=self.sample_size - n_new), dim=0
        )
        device = self.device
        assert str(device).startswith(
            "cuda"
        ), f"examples device is {device}, but should be cuda"
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)
        inp_timesteps = (
            torch.cat([rand_timesteps, old_timesteps], dim=0).detach().to(device)
        )

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(
            self.model,
            inp_imgs=inp_imgs,
            timesteps=inp_timesteps,
            sample_initializer=self.sample_initializer,
            steps=steps,
            step_size=step_size,
        )

        # Add new images to the buffer and remove old ones if needed
        self.buffered_examples = (
            list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.buffered_examples
        )
        self.buffered_examples = self.buffered_examples[: self.max_len]
        self.buffered_timesteps = (
            list(inp_timesteps.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.buffered_timesteps
        )
        self.buffered_timesteps = self.buffered_timesteps[: self.max_len]
        return inp_imgs, inp_timesteps

    @staticmethod
    def generate_samples(
        model,
        inp_imgs,
        inp_timesteps,
        sample_initializer: SampleInitializer = None,
        steps=60,
        step_size=10,
        diffusion_num_steps=100,
        return_img_per_step=False,
    ):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            inp_timesteps - Timesteps (noise levels) in the diffusion schedule. The output of this function is expected to have the same noise level.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # TODO: allow using a generator to perform amortized inference (initialize the sampled images)
        #    using a single forward path.
        if sample_initializer is not None:
            inp_imgs = sample_initializer(inp_imgs)

        # Generate the noise levels for the diffusion process
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_num_steps,
            beta_schedule="linear",
        )
        # TODO: how to get the correct alpha_t?
        alpha_t = noise_scheduler.alphas
        alpha_t = alpha_t.to(inp_imgs.device)
        inp_timesteps = inp_timesteps.long()

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

        # Loop over K (MCMC steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            energy = -model(inp_imgs, inp_timesteps)
            energy.sum().backward()
            inp_imgs.grad.data.clamp_(
                -0.03, 0.03
            )  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # TODO: divide by a scalar alpha_t
            # print("inp_imgs.shape", inp_imgs.shape)
            # print("inp_timesteps.shape", inp_timesteps.shape)
            multiplier = 1.0 / alpha_t[inp_timesteps]
            multiplier = multiplier.view(-1, 1, 1, 1)
            inp_imgs.data.mul_(multiplier)

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

    @staticmethod
    def generate_samples_0(
        model,
        inp_imgs,
        sample_initializer: SampleInitializer = None,
        steps=60,
        step_size=10,
        diffusion_num_steps=100,
        return_img_per_step=False,
    ):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            timesteps - Timesteps in the diffusion schedule.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # TODO: allow using a generator to perform amortized inference (initialize the sampled images)
        #    using a single forward path.
        if sample_initializer is not None:
            inp_imgs = sample_initializer(inp_imgs)

        # Generate the noise levels for the diffusion process
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_num_steps,
            beta_schedule="linear",
        )
        # TODO: how to get the correct alpha_t?
        alpha_t = noise_scheduler.alphas

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

        # Loop over T (diffusion steps)
        for t in list(range(diffusion_num_steps))[::-1]:
            timesteps = torch.ones(inp_imgs.shape[0], device=inp_imgs.device) * t
            # Loop over K (MCMC steps)
            for _ in range(steps):
                # Part 1: Add noise to the input.
                noise.normal_(0, 0.005)
                inp_imgs.data.add_(noise.data)
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

                # Part 2: calculate gradients for the current input.
                energy = -model(inp_imgs, timesteps)
                energy.sum().backward()
                inp_imgs.grad.data.clamp_(
                    -0.03, 0.03
                )  # For stabilizing and preventing too high gradients

                # Apply gradients to our current samples
                inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
                inp_imgs.grad.detach_()
                inp_imgs.grad.zero_()
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

                # TODO: divide by a scalar alpha_t
                inp_imgs.data.mul_(1.0 / alpha_t[t])

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
