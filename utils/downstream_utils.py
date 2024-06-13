import os
import gc

import MinkowskiEngine as ME
import torch
import pytorch_lightning as pl
from downstream.evaluate import evaluate
from downstream.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from downstream.lightning_trainer import LightningDownstream
from downstream.lightning_datamodule import DownstreamDataModule
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes


def downstream_train(config, resume_path=None, pretraining_path=None, random_seed=None,
                     epoch=None, train_logger=None, eval_logger=None):
    if resume_path:
        config["trainer"]["resume_path"] = resume_path
    if pretraining_path:
        config["trainer"]["pretraining_path"] = pretraining_path
    if random_seed:
        pl.seed_everything(random_seed)

    dm = DownstreamDataModule(config)
    model = make_model(config, config["trainer"]["pretraining_path"])
    if config["trainer"]["num_gpus"] > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    module = LightningDownstream(model, config)
    trainer = pl.Trainer(
        gpus=config["trainer"]["num_gpus"],
        accelerator="ddp",
        default_root_dir=config["working_dir"],
        checkpoint_callback=True,
        max_epochs=config["trainer"]["num_epochs"],
        plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["trainer"]["resume_path"],
        check_val_every_n_epoch=1,
        logger=train_logger,
    )
    print("Starting the training")
    trainer.fit(module, dm)

    torch.cuda.empty_cache()
    if config["trainer"]['num_gpus'] > 1:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/8375#issuecomment-878739663
        # Not fixed: https://github.com/pytorch/pytorch/issues/75097
        # TODO: Script hangs here but not on DGX. Need to figure out why
        torch.distributed.destroy_process_group()
        if not trainer.is_global_zero:
            return
    print("Training finished, now evaluating the results")
    del trainer
    del dm
    del module
    gc.collect()
    if config["dataset"]["name"].lower() == "nuscenes":
        phase = None
        if config["dataset"]["training"] in ("parametrize", "parametrizing"):
            phase = "verifying"
        elif config["dataset"]["training"] == "miniset":
            phase = "mini_val"
        else:
            phase = "val"

        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["trainer"]["num_threads"]
        )
    elif config["dataset"]["name"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["trainer"]["num_threads"]
        )
    evaluate(model.to(0), val_dataloader, config,
             epoch=epoch, logger=eval_logger)
