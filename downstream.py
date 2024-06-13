import argparse
from pathlib import Path

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
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )
    parser.add_argument(
        "--extra_tag", type=str, default='default', help='Extra tag output directory'
    )
    parser.add_argument(
        "--random_seed", type=int, default=2022, help='Set random seed'
    )
    args = parser.parse_args()

    config = generate_config(args.cfg_file, extra_tag=args.extra_tag, random_seed=args.random_seed)
    print_config(config)
    # Remove 'config' and 'xxxx.yaml'
    working_dir = Path('output') / Path('/'.join(args.cfg_file.split('/')[1:-1])) / Path(args.cfg_file).stem / config["extra_tag"]
    config["working_dir"] = working_dir

    downstream_train(config, resume_path=args.resume_path, pretraining_path=args.pretraining_path, random_seed=args.random_seed)


if __name__ == "__main__":
    main()
