import argparse
from pathlib import Path

from pytorch_lightning.loggers import WandbLogger
from utils.read_config import generate_config, print_config
from utils.downstream_utils import downstream_train


def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--extra_tag", type=str, default="default", help="Extra tag for output"
    )
    parser.add_argument(
        "--mod_cfg_file", type=str, default=None, help="Specify the config file to modify"
    )
    parser.add_argument(
        "--disable_wandb", default=False, action='store_true', help="Disable wandb reporting"
    )
    # TODO: Add support for both nuscenes and kitti evaluation
    parser.add_argument("--eval_type", required=True, type=str, choices=["finetuning_nuscenes", "linear_probe_nuscenes", "finetuning_semkitti", "linear_probe_semkitti"])
    parser.add_argument(
        "--random_seed", type=int, default=2022, help='Set random seed'
    )
    args = parser.parse_args()
    eval_type = args.eval_type

    config = generate_config(args.cfg_file, mod_file=args.mod_cfg_file, extra_tag=args.extra_tag, random_seed=args.random_seed)
    # Remove 'config' and 'xxxx.yaml'
    working_dir = Path('output') / Path('/'.join(args.cfg_file.split('/')[1:-1])) / Path(args.cfg_file).stem / config["extra_tag"]
    config["working_dir"] = working_dir
    if not config["eval"].get(eval_type, {}).get("enabled"):
        print('Not running finetuning')
        return

    checkpoint_path = working_dir / "model.pt"
    assert checkpoint_path.exists()
    downstream_config = generate_config(config["eval"][eval_type]["cfg_file"], extra_tag=args.extra_tag)
    downstream_config["working_dir"] = working_dir / eval_type
    downstream_config["experiment"] = config.get("experiment")
    downstream_config["trainer"]["dataset_skip_step"] = config["eval"][eval_type]["dataset_skip_step"]
    
    if eval_type == "finetuning_nuscenes" or eval_type == "finetuning_semkitti":
        downstream_config["model"]["freeze_layers"] = False
    elif eval_type == "linear_probe_nuscenes" or eval_type == "linear_probe_semkitti":
        downstream_config["model"]["freeze_layers"] = True
    else:
        raise NotImplemented
    downstream_config["log_groups"] = [eval_type]
    print_config(downstream_config)

    wandb_logger = None
    if config.get("wandb", {}).get("enabled") and not args.disable_wandb:
        wandb_name = Path(args.cfg_file).stem
        wandb_logger = WandbLogger(name=wandb_name, config=downstream_config,
                                   project=config["wandb"]["project"],
                                   entity=config["wandb"]["entity"],
                                   group=f'{wandb_name}-{config["extra_tag"]}',
                                   job_type=eval_type)

    downstream_train(downstream_config, resume_path=None, pretraining_path=checkpoint_path,
                     random_seed=args.random_seed, train_logger=wandb_logger, eval_logger=wandb_logger)


if __name__ == "__main__":
    main()
