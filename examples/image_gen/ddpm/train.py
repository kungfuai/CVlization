import json
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from ddim import *
from ddpm_torch import *
from functools import partial
from torch.distributed.elastic.multiprocessing import errors
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.optim import Adam, lr_scheduler


def train(rank=0, args=None, temp_dir=""):
    distributed = args.distributed

    def logger(msg, **kwargs):
        if not distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    root = os.path.expanduser(args.root)
    if args.config_path is None:
        args.config_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(args.config_path, "r") as f:
        meta_config = json.load(f)
    exp_name = os.path.basename(args.config_path)[:-5]

    # dataset basic info
    dataset = meta_config.get("dataset", args.dataset)
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    # set seed for RNGs
    seed = meta_config.get("seed", args.seed)
    seed_all(seed)

    # extract training-specific hyperparameters
    gettr = partial(get_param, obj_1=meta_config.get("train", {}), obj_2=args)
    train_config = ConfigDict(**{
        k: gettr(k) for k in (
            "batch_size", "beta1", "beta2", "lr", "epochs", "grad_norm", "warmup",
            "chkpt_intv", "image_intv", "num_samples", "use_ema", "ema_decay")})
    train_config.batch_size //= args.num_accum
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # extract diffusion-specific hyperparameters
    getdif = partial(get_param, obj_1=meta_config.get("diffusion", {}), obj_2=args)
    diffusion_config = ConfigDict(**{
        k: getdif(k) for k in (
            "beta_schedule", "beta_start", "beta_end", "timesteps",
            "model_mean_type", "model_var_type", "loss_type")})

    betas = get_beta_schedule(
        diffusion_config.beta_schedule, beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end, timesteps=diffusion_config.timesteps)
    diffusion = GaussianDiffusion(betas=betas, **diffusion_config)

    # extract model-specific hyperparameters
    out_channels = 2 * in_channels if diffusion_config.model_var_type == "learned" else in_channels
    model_config = meta_config["model"]
    block_size = model_config.pop("block_size", args.block_size)
    model_config["in_channels"] = in_channels * block_size ** 2
    model_config["out_channels"] = out_channels * block_size ** 2
    _model = UNet(**model_config)

    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        _model = ModelWrapper(_model, pre_transform, post_transform)

    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()

        if args.rigid_launch:
            # launched by torch.multiprocessing.spawn
            # share information and initialize the distributed package via shared file-system (FileStore)
            # adapted from https://github.com/NVlabs/stylegan2-ada-pytorch
            # currently, this only supports single-node training
            assert temp_dir, "Temporary directory cannot be empty!"
            init_method = f"file://{os.path.join(os.path.abspath(temp_dir), '.torch_distributed_init')}"
            dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=args.num_gpus)
            local_rank = rank
            os.environ["WORLD_SIZE"] = str(args.num_gpus)
            os.environ["LOCAL_RANK"] = str(rank)
        else:
            # launched by either torch.distributed.elastic (single-node) or Slurm srun command (multi-node)
            # elastic launch with C10d rendezvous backend by default uses TCPStore
            # initialize with environment variables for maximum customizability
            world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
            # global process id across all node(s)
            local_world_size = (int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or
                                int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or
                                torch.cuda.device_count())
            # local device id on a single node
            local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
            args.num_gpus = world_size or local_world_size
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))

        logger(f"Using distributed training with {args.num_gpus} GPU(s).")
        torch.cuda.set_device(local_rank)
        _model.cuda()
        model = DDP(_model, device_ids=[local_rank, ])
        train_device = torch.device(f"cuda:{local_rank}")

    else:
        rank = local_rank = 0
        model = _model.to(train_device)

    is_leader = rank == 0  # rank 0: leader in the process group

    logger(f"Dataset: {dataset}")
    logger(
        f"Effective batch-size is {train_config.batch_size} * {args.num_accum}"
        f" = {train_config.batch_size * args.num_accum}.")

    # PyTorch's implementation of Adam differs slightly from TensorFlow in that
    # the former follows Algorithm 1 as described in the paper by Kingma & Ba (2015) [1]
    # while the latter adopts an alternative approach mentioned just before Section 2.1
    # see also https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64-L69
    optimizer = Adam(model.parameters(), lr=train_config.lr, betas=(train_config.beta1, train_config.beta2))
    # lr_lambda is used to calculate the learning rate multiplicative factor at timestep t (starting from 0)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / train_config.warmup, 1.0)
    ) if train_config.warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=train_config.batch_size, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    if args.dry_run:
        logger("This is a dry run.")
        args.chkpt_intv = 1
        args.image_intv = 1

    chkpt_dir = os.path.join(args.chkpt_dir, exp_name)
    chkpt_path = os.path.join(chkpt_dir, args.chkpt_name or f"{exp_name}.pt")
    chkpt_intv = args.chkpt_intv
    logger(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}", end=" ")
    logger(f"every {chkpt_intv} epoch(s)")

    image_dir = os.path.join(args.image_dir, "train", exp_name)
    logger(f"Generated images (x{train_config.num_samples}) will be saved to {os.path.abspath(image_dir)}", end=" ")
    logger(f"every {train_config.image_intv} epoch(s)")

    if is_leader:
        model_config["block_size"] = block_size
        hps = {
            "dataset": dataset,
            "seed": seed,
            "diffusion": diffusion_config,
            "model": model_config,
            "train": train_config
        }
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        # keep a record of hyperparameter settings used for this experiment run
        with open(os.path.join(chkpt_dir, f"exp_{timestamp}.info"), "w") as f:
            json.dump(hps, f, indent=2)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=train_config.epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        num_accum=args.num_accum,
        use_ema=train_config.use_ema,
        grad_norm=train_config.grad_norm,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=train_config.image_intv,
        num_samples=train_config.num_samples,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed,
        dry_run=args.dry_run
    )

    if args.use_ddim:
        subsequence = get_selection_schedule(
            args.skip_schedule, size=args.subseq_size, timesteps=diffusion_config.timesteps)
        diffusion_eval = DDIM.from_ddpm(diffusion, eta=0., subsequence=subsequence)
    else:
        diffusion_eval = diffusion

    if args.eval:
        evaluator = Evaluator(
            dataset=dataset,
            diffusion=diffusion_eval,
            eval_batch_size=args.eval_batch_size,
            eval_total_size=args.eval_total_size,
            device=eval_device
        )
    else:
        evaluator = None

    # in the case of distributed training, resume should always be turned on
    resume = args.resume or distributed
    if resume:
        try:
            map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
            _chkpt_path = args.chkpt_path or chkpt_path
            trainer.load_checkpoint(_chkpt_path, map_location=map_location)
        except FileNotFoundError:
            logger("Checkpoint file does not exist!")
            logger("Starting from scratch...")

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir)


@errors.record
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, help="path to the configuration file")
    parser.add_argument("--exp-name", type=str, help="name of the current experiment run")
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=50, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=1, type=int, help="number of mini-batches before an update")
    parser.add_argument("--block-size", default=1, type=int, help="block size used for pixel shuffle")
    parser.add_argument("--timesteps", default=1000, type=int, help="number of diffusion steps")
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.0001, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)  # noqa
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images", type=str)
    parser.add_argument("--image-intv", default=10, type=int)
    parser.add_argument("--num-samples", default=64, type=int, help="number of images to sample and save")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-name", default="", type=str)
    parser.add_argument("--chkpt-intv", default=120, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--chkpt-path", default="", type=str, help="checkpoint path used to resume training")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--eval-total-size", default=50000, type=int)
    parser.add_argument("--eval-batch-size", default=256, type=int)
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")
    parser.add_argument("--use-ddim", action="store_true", help="whether to use DDIM sampler for evaluation")
    parser.add_argument("--skip-schedule", choices=["linear", "quadratic"], default="linear", type=str)
    parser.add_argument("--subseq-size", default=50, type=int)
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")
    parser.add_argument("--rigid-launch", action="store_true", help="whether to use torch multiprocessing spawn")
    parser.add_argument("--num-gpus", default=1, type=int, help="number of gpus for distributed training")
    parser.add_argument("--dry-run", action="store_true", help="test-run till the first model update completes")

    args = parser.parse_args()

    if args.distributed and args.rigid_launch:
        mp.set_start_method("spawn")
        with tempfile.TemporaryDirectory() as temp_dir:
            mp.spawn(train, args=(args, temp_dir), nprocs=args.num_gpus)
    else:
        """
        As opposed to the case of rigid launch, distributed training now:
        (*: elastic launch only; **: Slurm srun only)
         *1. handles failures by restarting all the workers 
         *2.1 assigns RANK and WORLD_SIZE automatically
        **2.2 sets MASTER_ADDR & MASTER_PORT manually beforehand via environment variables
         *3. allows for number of nodes change
          4. uses TCP initialization by default
        **5. supports multi-node training
        """
        train(args=args)


if __name__ == "__main__":
    main()
