from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

# Remove a few layers to fit on 24GB
tokenizer = VideoTokenizer(
    image_size = 256,
    init_dim = 64,
    max_dim = 512,
    codebook_size = 1024,
    layers = (
        'residual',
        'compress_space',
        ('consecutive_residual', 1),
        # 'compress_space',
        # ('consecutive_residual', 2),
        # 'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space',
        'compress_time',
        ('consecutive_residual', 1),
        # 'compress_time',
        # ('consecutive_residual', 2),
        'attend_time',
    )
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder = 'data/flying_mnist/train',     # folder of either videos or images, depending on setting below
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 1,
    grad_accum_every = 16,
    learning_rate = 2e-5,
    num_train_steps = 1_000_000,
    use_wandb_tracking=True,
)

with trainer.trackers(project_name = 'magvit2'):
    trainer.train()