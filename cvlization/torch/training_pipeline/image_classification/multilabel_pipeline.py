import torchvision
from torchvision import transforms
import torch
from typing import List
from dataclasses import field, dataclass
from lightning.pytorch.loggers import MLFlowLogger
from .lightning_kornia import MultiLabelImageClassifier, ImageClassifierCallback
from .transform import TransformedDataset


class MultiLabelClassification:
    @dataclass
    class Config:
        # model
        label_fields: List[str]
        num_classes: int = 5
        model_name: str = "resnet101"
        pretrained: bool = True
        use_color_hist: bool = False
        color_hist_num_bins: int = 32
        color_hist_bandwidth: int = 0.001

        # model checkpoint to resume from
        checkpoint_model_path: str = None
        checkpoint_experiment_name: str = None
        checkpoint_run_name: str = None

        # training loop
        epochs: int = 100
        train_steps_per_epoch: int = None
        val_steps_per_epoch: int = None
        early_stopping: bool = True
        early_stopping_metric = "val_ap"
        early_stopping_patience = 10
        early_stopping_min_delta = 0.005
        
        # dataset
        filter_no_visible_bin: bool = False
        val_proportion: float = 0.25 
        train_set_to_exclude_prop: float = 0

        # optimizer
        lr: float = 0.001
        lr_scheduling: bool = False

        # data loading
        batch_size: int = 16  # 16 for resnet50, 64 for resnet18
        num_workers: int = cpu_count()

        # device
        gpus: List[int] = field(default_factory=lambda: [3])
        accelerator: str = "gpu"

        # experiment tracking
        experiment_name: str = "residential"
        run_name: str = "test_resnet101"
        tracking_uri: str = "./mlruns"
        lightning_root_dir: str = "./lightning_logs"

        # transforms
        rotation_angle: int = 30
        random_shift: float = 0.2
        random_scale_min: float = 0.8
        random_scale_max: float = 1.2
        gaussian_blur_kernel_size: int = 3
        gaussian_blur_sigma_max: float = 0.001
        erase_p: float = 0
        adjust_sharpness_factor: float = 1.5
        adjust_sharpness_p: float = 0
        crop_scale_min: float = 0
        crop_scale_max: float = 0

    def __init__(self, config: Config):
        self._config = config

    def fit(self, dataset_builder):
        pl_model = self._create_model(dataset_builder)
        self._setup_experiment_tracker()
        trainer = self._create_trainer(dataset_builder)
        train_dataloader = self._create_training_dataloader(dataset_builder)
        val_dataloader = self._create_validation_dataloader(dataset_builder)
        trainer.fit(pl_model, train_dataloader, val_dataloader)

    def _create_model(self, dataset_builder):
        model_constructor = getattr(torchvision.models, self._config.model_name)
        model = model_constructor(pretrained=self._config.pretrained)
        # model.fc = nn.Linear(model.fc.in_features, self._config.num_classes)
        pl_model = MultiLabelImageClassifier(
            model=model,
            num_classes=self._config.num_classes,
            lr=self._config.lr,
            lr_scheduling=self._config.lr_scheduling,
            use_color_hist=self._config.use_color_hist,
            color_hist_num_bins=self._config.color_hist_num_bins,
            color_hist_bandwidth=self._config.color_hist_bandwidth,
        )
        pl_model.loss = torch.nn.BCEWithLogitsLoss()
        return pl_model

    def _create_trainer(self, dataset_builder):
        class_labels = dataset_builder.training_dataset().label_fields
        callbacks = [
            ImageClassifierCallback(class_labels=class_labels),
            ModelCheckpoint(
                dirpath=f"models/{self._config.experiment_name}/{self._config.run_name}/",
                # filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                filename="model",
                monitor="val_ap",
                mode="max",
                save_top_k=1,
                every_n_epochs=1,
            ),
        ]
        if self._config.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self._config.early_stopping_metric,
                    min_delta=self._config.early_stopping_min_delta,
                    patience=self._config.early_stopping_patience,
                    verbose=False,
                    mode="max",
                )
            )
        trainer = Trainer(
            default_root_dir=self._config.lightning_root_dir,
            deterministic=True,
            accelerator=self._config.accelerator,
            gpus=self._config.gpus,
            limit_train_batches=self._config.train_steps_per_epoch or 1.0,
            limit_val_batches=self._config.val_steps_per_epoch or 1.0,
            max_epochs=self._config.epochs,
            logger=self._experiment_tracker,
            callbacks=callbacks,
        )
        return trainer

    def _setup_experiment_tracker(self):
        self._experiment_tracker = experiment_tracker = MLFlowLogger(
            experiment_name=self._config.experiment_name,
            run_name=self._config.run_name,
            tracking_uri=self._config.tracking_uri,
        )

        for k, v in self._config.__dict__.items():
            experiment_tracker.experiment.log_param(
                run_id=experiment_tracker.run_id, key=k, value=v
            )
        return self._experiment_tracker

    def _transform_training_dataset(self, dataset):
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(self._config.rotation_angle),
                transforms.RandomAffine(
                    degrees=self._config.rotation_angle,
                    translate=(self._config.random_shift, self._config.random_shift),
                    scale=(
                        self._config.random_scale_min,
                        self._config.random_scale_max,
                    ),
                ),
                transforms.GaussianBlur(
                    self._config.gaussian_blur_kernel_size,
                    (0.001, self._config.gaussian_blur_sigma_max),
                ),
                transforms.RandomErasing(self._config.erase_p),
                transforms.RandomAdjustSharpness(
                    self._config.adjust_sharpness_factor,
                    self._config.adjust_sharpness_p,
                ),
                transforms.RandomResizedCrop(
                    [480, 704],
                    (self._config.crop_scale_min, self._config.crop_scale_max),
                ),
            ]
        )
        return TransformedDataset(dataset, transform_train)

    def _transform_validation_dataset(self, dataset):
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return TransformedDataset(dataset, transform_val)

    def _create_training_dataloader(self, dataset_builder):
        train_ds = self._transform_training_dataset(dataset_builder.training_dataset())
        return DataLoader(
            train_ds,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
        )

    def _create_validation_dataloader(self, dataset_builder):
        val_ds = self._transform_validation_dataset(
            dataset_builder.validation_dataset()
        )
        return DataLoader(
            val_ds,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
        )