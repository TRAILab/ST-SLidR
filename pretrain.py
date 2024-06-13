import os
import argparse
from pathlib import Path

import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl

from utils.read_config import generate_config, print_config
from utils.checkpoint_utils import PeriodicCheckpoint
from pretrain.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule import PretrainDataModule


def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--extra_tag", type=str, default='default', help='Extra tag output directory'
    )
    parser.add_argument(
        "--mod_cfg_file", type=str, default=None, help="Specify the config file to modify"
    )
    parser.add_argument(
        "--disable_wandb", default=False, action='store_true', help="Disable wandb reporting"
    )
    parser.add_argument(
        "--random_seed", type=int, default=2022, help='Set random seed'
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file, mod_file=args.mod_cfg_file, extra_tag=args.extra_tag, random_seed=args.random_seed)
    # Remove 'config' and 'xxxx.yaml'
    working_dir = Path('output') / Path('/'.join(args.cfg_file.split('/')[1:-1])) / Path(args.cfg_file).stem / config["extra_tag"]
    config["working_dir"] = working_dir

    if args.resume_path:
        config["trainer"]["resume_path"] = args.resume_path

    if args.random_seed:
        pl.seed_everything(args.random_seed)

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print_config(config)

    dm = PretrainDataModule(config)
    model_points, model_images = make_model(config)
    if config["trainer"]["num_gpus"] > 1:
        model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    assert config["model"]["pointcloud"]["backbone"] == "minkunet"
    module = LightningPretrain(model_points, model_images, config)
    wandb_logger = None
    if config.get("wandb", {}).get("enabled") and not args.disable_wandb:
        wandb_name = Path(args.cfg_file).stem
        wandb_logger = WandbLogger(name=wandb_name, config=config,
                                   project=config["wandb"]["project"],
                                   entity=config["wandb"]["entity"],
                                   group=f'{wandb_name}-{config["extra_tag"]}',
                                   job_type="pretrain")

    callbacks = []

    checkpoint_dir = working_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    periodic_checkpoint = PeriodicCheckpoint(config["trainer"].get('checkpoint_every_n_epochs', 0), dirpath=checkpoint_dir)
    callbacks.append(periodic_checkpoint)

    trainer = pl.Trainer(
        gpus=config["trainer"]["num_gpus"],
        accelerator="ddp",
        default_root_dir=working_dir,
        checkpoint_callback=True,
        max_epochs=config["trainer"]["num_epochs"],
        plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["trainer"]["resume_path"],
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    print("Starting the training")
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
