data_root = '/data/data'
data = dict(type='SAM', root='images', image_list_txt='part0.txt', transform='default_train', load_vae_feat=True)
image_size = 256  # the generated image resolution
train_batch_size = 16
eval_batch_size = 2
use_fsdp=False   # if use FSDP mode

# model setting
model = 'PixArt_XL_2'
window_block_indexes=[]
window_size=0
use_rel_pos=False
fp32_attention = True
aspect_ratio_type = None         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_1024]
multi_scale = False     # if use multiscale dataset model training
lewei_scale = 1.0
model_max_length = 120
lora_rank = 4
# training setting
num_workers=4
train_sampling_steps = 1000
eval_sampling_steps = 250

num_epochs = 80
gradient_accumulation_steps = 1
grad_checkpointing = False
gc_step = 1
gradient_clip = 1.0
auto_lr = dict(rule='sqrt')

# we use different weight decay with the official implementation since it results better result
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=1000)

save_image_epochs = 1
save_model_epochs = 20
save_model_steps=1000000

sample_posterior = True
mixed_precision = 'fp16'
scale_factor = 0.18215
ema_rate = 0.9999
tensorboard_mox_interval = 50
log_interval = 50
cfg_scale = 4
mask_type='null'
num_group_tokens=0
mask_loss_coef=0.
load_mask_index=False    # load prepared mask_type index
# load model settings
# vae_pretrained = "/cache/pretrained_models/sd-vae-ft-ema"
vae_pretrained = "stabilityai/sd-vae-ft-mse"
load_from = None
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
snr_loss=False

# work dir settings
work_dir = '/cache/exps/'
s3_work_dir = None

seed = 43
# latents_input_file = "data/latents/flying_mnist__sd-vae-ft-mse_latents_32frames_train.npy"
latents_input_file = "data/latents/flying_mnist_11k__sd-vae-ft-mse_latents_32frames_train.npy"