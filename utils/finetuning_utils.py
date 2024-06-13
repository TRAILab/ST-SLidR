from pathlib import Path

from utils.read_config import generate_config
from utils.downstream_utils import downstream_train


def finetune_eval(checkpoint_path, downstream_config_file, default_root_dir, extra_tag, dataset_skip_step=None, logger=None):
    """
    Run finetuning on models in checkpoints_dir with dataset_skip_set list.

    Args:
    checkpoint_path: Path object to checkpoint file. If a file is specified, only evaluate file.
    If a directory is specified, evaluate all checkpoints in dir
    downstream_config_file: Config file path for downstream task
    default_root_dir: Path for training output
    dataset_skip_step: 1 / dataset_skip_step for finetuning percentage
    """
    assert isinstance(checkpoint_path, Path)
    assert isinstance(default_root_dir, Path)

    downstream_config = generate_config(downstream_config_file, extra_tag=extra_tag)
    downstream_config['working_dir'] = default_root_dir / 'finetuning'
    if dataset_skip_step is not None:
        downstream_config['dataset_skip_step'] = dataset_skip_step

    if logger is not None:
        logger.log_hyperparams({'finetuning_config': downstream_config})

    if checkpoint_path.is_file():
        print(f'Training: {checkpoint_path}')
        downstream_config['finetuning_epoch'] = 0
        downstream_train(downstream_config, resume_path=None, pretraining_path=checkpoint_path,
                         eval_logger=logger, log_groups=['finetune'])
    elif checkpoint_path.is_dir():
        for checkpoint_file in sorted(checkpoint_path.iterdir()):
            print(f'Training: {checkpoint_file}')
            downstream_config['finetuning_epoch'] = int(checkpoint_file.stem.split('-')[-1])
            downstream_train(downstream_config, resume_path=None, pretraining_path=checkpoint_file,
                             eval_logger=logger, log_groups=['finetune'])
    else:
        raise NotImplementedError
