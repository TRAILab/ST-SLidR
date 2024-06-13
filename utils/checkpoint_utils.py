from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, dirpath=None):
        super().__init__(dirpath=dirpath)
        self.every = every

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Epoch starts at 0, increment by 1. Also save last epoch.
        curr_epoch = pl_module.current_epoch + 1
        if (self.every > 0 and curr_epoch % self.every == 0) or curr_epoch == pl_module._config["trainer"]["num_epochs"]:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{curr_epoch:03}.pt"
            pl_module.save(current)
