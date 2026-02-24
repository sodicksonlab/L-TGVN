import torch
import warnings
import argparse
import os
import math
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
)
from lightning.pytorch.loggers import WandbLogger
from torch.optim import Adam, AdamW
from omegaconf import OmegaConf
from typing import Optional
from src.models import VN, TGVN
from src.loss import (
    SSIMLoss, EASSIMLoss, EAL1Loss, ValMetrics
)
from src.pl_data_module import TGVNDataModule


# Suppress irrelevant warnings
warnings.filterwarnings("ignore",
                        ".*returns a result that is inconsistent.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")
warnings.filterwarnings("ignore",
                        ".*due to potential conflicts with other packages.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore",
                        ".*functools.partial will be a method descriptor.*")
warnings.filterwarnings("ignore", ".*No device id is provided via.*")
warnings.filterwarnings("ignore", ".*FutureWarning:*")


def custom_format_checkpoint_name(original_format):
    """Custom checkpoint naming function to remove 'epoch=' prefix."""
    def format_checkpoint_name(metrics, filename=None, ver=None):
        name = original_format(metrics, filename, ver)
        name = name.replace('epoch=', '')
        return name
    return format_checkpoint_name


class OptimizerStepProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # Add optimizer step count
        items["opt_step"] = trainer.global_step
        return items


class LinearWarmupExponentialDecayScheduler:
    """
    Custom scheduler that combines optional linear warmup
    with step-based exponential decay.
    If warmup_epochs > 0:
        - Linear warmup for warmup_epochs
        - Exponential decay based on epoch after warmup
    If warmup_epochs is set to 0:
        - Exponential decay based on epoch from the beginning
    `epoch` is continuous, not discrete.
    """
    def __init__(
            self,
            optimizer,
            warmup_epochs: int,
            lr_gamma: float,
            steps_per_epoch: int,
     ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.lr_gamma = lr_gamma
        self.steps_per_epoch = steps_per_epoch

        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.initial_lr = optimizer.param_groups[0]['lr']
        self._step = 0

    def step(self):
        self._step += 1
        if self.warmup_epochs > 0 and self._step <= self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (self._step / self.warmup_steps)
        else:
            # Exponential decay phase (every step)
            if self.warmup_epochs > 0:
                # Decay based on steps after warmup
                decay_steps = self._step - self.warmup_steps
            else:
                # Decay based on total steps from beginning
                decay_steps = self._step
            lr = self.initial_lr * (
                self.lr_gamma ** (decay_steps / self.steps_per_epoch)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'_step': self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict['_step']


class TGVNLightning(L.LightningModule):
    def __init__(
        self,
        model_type: str = 'TGVN',
        num_cascades: int = 12,
        sens_chans: int = 18,
        Phi_chans: int = 25,
        H_chans: int = 25,
        loss_function: str = 'ssim',
        edge_weight: Optional[float] = None,
        learning_rate: float = 3e-4,
        lr_gamma: float = 0.99,
        warmup_epochs: int = 5,
        weight_decay: Optional[float] = None,
    ):
        super().__init__()
        self.model_type = model_type.upper()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_gamma = lr_gamma
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        if self.model_type == 'TGVN':
            self.model = TGVN(
                num_cascades=num_cascades,
                sens_chans=sens_chans,
                Phi_chans=Phi_chans,
                H_chans=H_chans
            )
        elif self.model_type == 'VN':
            self.model = VN(
                num_cascades=num_cascades,
                sens_chans=sens_chans,
                Phi_chans=Phi_chans
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}.")

        if loss_function == 'ssim':
            self.loss_fn = SSIMLoss()
        elif loss_function == 'l1':
            l1_loss = torch.nn.L1Loss()
            self.loss_fn = lambda pred, target, min_target=None, \
                max_target=None: l1_loss(pred, target)
        elif loss_function == 'eassim':
            assert edge_weight is not None, (
                "Edge weight must be provided for EASSIMLoss."
            )
            if edge_weight <= 0:
                raise ValueError("Edge weight must be positive.")
            self.loss_fn = EASSIMLoss(
                edge_weight=edge_weight  # type: ignore
            )
        elif loss_function == 'eal1':
            assert edge_weight is not None, (
                "Edge weight must be provided for EASSIMLoss."
            )
            if edge_weight <= 0:
                raise ValueError("Edge weight must be positive.")
            self.loss_fn = EAL1Loss(
                edge_weight=edge_weight  # type: ignore
            )
        else:
            raise ValueError(
                f"Unsupported loss function: {loss_function}."
            )

        # Prints SSIM, L1Loss and EdgeAwareLoss in validation and test
        self.debug_fn = ValMetrics()

    def on_save_checkpoint(self, checkpoint):
        if (hasattr(self.logger, 'experiment') and
                hasattr(self.logger.experiment, 'id')):  # type: ignore
            checkpoint['wandb_run_id'] = self.logger.experiment.id  # type: ignore  # noqa: E501

    def forward(self, *args, **kwargs):
        # Generic pass-through so both VN and TGVN signatures work
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        kspace = batch.kspace
        mask = batch.mask
        prior = batch.prior
        target = batch.target
        recon_size = batch.recon_size
        min_target = batch.min_target
        max_target = batch.max_target
        acceleration = batch.acceleration
        self.log(
            'acceleration', acceleration, on_step=True,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )

        if self.model_type == 'TGVN':
            pred = self.model(
                kspace, mask, prior, recon_size

            )
        else:
            pred = self.model(
                kspace, mask, recon_size
            )

        loss_output = self.loss_fn(pred, target, min_target, max_target)
        if isinstance(loss_output, tuple):  # EASSIMLoss or EAL1Loss
            loss, main_loss, edge_loss = loss_output
            self.log(
                'train_loss', loss, on_step=True,
                on_epoch=True, prog_bar=True,
                sync_dist=True, batch_size=target.shape[0]
            )
            self.log(
                'train_main_loss', main_loss, on_step=True,
                on_epoch=True, prog_bar=True,
                sync_dist=True, batch_size=target.shape[0]
            )
            self.log(
                'train_edge_loss', edge_loss, on_step=True,
                on_epoch=True, prog_bar=True,
                sync_dist=True, batch_size=target.shape[0]
            )
        else:
            loss = loss_output
            self.log(
                'train_loss', loss, on_step=True,
                on_epoch=True, prog_bar=True,
                sync_dist=True, batch_size=target.shape[0]
            )
        return loss

    def validation_step(self, batch, batch_idx):
        kspace = batch.kspace
        mask = batch.mask
        prior = batch.prior
        target = batch.target
        recon_size = batch.recon_size
        min_target = batch.min_target
        max_target = batch.max_target
        acceleration = batch.acceleration
        self.log(
            'acceleration', acceleration, on_step=True,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )

        if self.model_type == 'TGVN':
            pred = self.model(
                kspace, mask, prior, recon_size

            )
        else:
            pred = self.model(
                kspace, mask, recon_size
            )
        batch_ssim, batch_l1, batch_edge = self.debug_fn(
            pred, target, min_target, max_target
        )
        self.log(
            'val_ssim', batch_ssim, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )
        self.log(
            'val_l1_loss', batch_l1, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )
        self.log(
            'val_edge_loss', batch_edge, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )

    def test_step(self, batch, batch_idx):
        kspace = batch.kspace
        mask = batch.mask
        prior = batch.prior
        target = batch.target
        recon_size = batch.recon_size
        min_target = batch.min_target
        max_target = batch.max_target
        acceleration = batch.acceleration
        self.log(
            'acceleration', acceleration, on_step=True,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )

        if self.model_type == 'TGVN':
            pred = self.model(
                kspace, mask, prior, recon_size

            )
        else:
            pred = self.model(
                kspace, mask, recon_size
            )
        batch_ssim, batch_l1, batch_edge = self.debug_fn(
            pred, target, min_target, max_target
        )
        self.log(
            'val_ssim', batch_ssim, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )
        self.log(
            'val_l1_loss', batch_l1, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )
        self.log(
            'val_edge_loss', batch_edge, on_step=False,
            on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=target.shape[0]
        )

    def configure_optimizers(self):
        if self.weight_decay is not None:
            optimizer = AdamW(
                self.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            if self.trainer.is_global_zero:
                print(f"Using AdamW with weight decay: {self.weight_decay}")
        else:
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
            if self.trainer.is_global_zero:
                print("Using Adam without weight decay")
        # Get total training samples and calculate global steps
        datamodule = self.trainer.datamodule  # type: ignore
        total_train_samples = len(datamodule.train_dataloader().dataset)
        batch_size_per_gpu = datamodule.train_batch_size
        world_size = self.trainer.world_size
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        # Effective batch size accounts for gradient accumulation
        effective_batch_size = (
            batch_size_per_gpu * world_size * accumulate_grad_batches
        )

        # Global steps per epoch = number of optimizer steps
        global_steps_per_epoch = math.floor(
            total_train_samples / effective_batch_size
        )

        if self.trainer.is_global_zero:
            print("Warmup calculation:")
            print(f"Total training samples: {total_train_samples}")
            print(f"Batch size per GPU: {batch_size_per_gpu}")
            print(f"World size (total GPUs): {world_size}")
            print(f"Gradient accumulation steps: {accumulate_grad_batches}")
            print(f"Effective global batch size: {effective_batch_size}")
            print(f"Global steps per epoch: {global_steps_per_epoch}")
            warmup_steps = self.warmup_epochs * global_steps_per_epoch
            print(f"Warmup will last {self.warmup_epochs} epochs = "
                  f"{warmup_steps} global steps")

        scheduler = LinearWarmupExponentialDecayScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            lr_gamma=self.lr_gamma,
            steps_per_epoch=global_steps_per_epoch
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    checkpoint_path = None
    if config.checkpoint.resume_from is not None:
        checkpoint_path = os.path.join(
            config.checkpoint.dirpath, config.checkpoint.resume_from
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

    try:
        from lightning.pytorch.utilities import rank_zero_only
        if rank_zero_only.rank == 0:
            print("Configuration:")
            print(OmegaConf.to_yaml(config))
    except (ImportError, AttributeError):
        print("Configuration:")
        print(OmegaConf.to_yaml(config))

    data_module = TGVNDataModule(
        train_batch_size=config.data.train_batch_size,
        val_batch_size=config.data.val_batch_size,
        workers=config.data.num_workers,
        buffer_size=config.data.buffer_size,
        target_acceleration=config.data.target_acceleration,
        center_fraction=config.data.center_fraction,
    )

    model = TGVNLightning(
        model_type=config.model.model_type,
        num_cascades=config.model.num_cascades,
        sens_chans=config.model.sens_chans,
        Phi_chans=config.model.Phi_chans,
        H_chans=config.model.get('H_chans', None),
        loss_function=config.training.loss_function,
        edge_weight=config.training.get('edge_weight', None),
        learning_rate=config.optimization.learning_rate,
        lr_gamma=config.optimization.lr_gamma,
        warmup_epochs=config.optimization.warmup_epochs,
        weight_decay=config.optimization.get('weight_decay', None),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=f"{config.model.model_type}_{{epoch}}",
        save_top_k=config.checkpoint.save_top_k,
        every_n_epochs=config.checkpoint.every_n_epochs,
        save_on_train_epoch_end=config.checkpoint.save_on_train_epoch_end,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        auto_insert_metric_name=False,
    )
    checkpoint_callback.format_checkpoint_name = custom_format_checkpoint_name(
        checkpoint_callback.format_checkpoint_name
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup WandB logger
    wandb_logger_kwargs = {
        'project': config.logger.project,
        'entity': config.logger.entity,
        'name': config.logger.name,
        'save_dir': config.logger.save_dir,
    }

    wandb_logger = WandbLogger(**wandb_logger_kwargs)
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        num_nodes=config.hardware.num_nodes,
        strategy=config.hardware.strategy,
        callbacks=[
            checkpoint_callback, lr_monitor, OptimizerStepProgressBar()
        ],
        logger=wandb_logger,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    trainer.fit(model, data_module, ckpt_path=checkpoint_path)


if __name__ == '__main__':
    main()
