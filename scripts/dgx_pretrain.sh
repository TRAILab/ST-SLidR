#!/bin/bash
#SBATCH --job-name=pretrain    # Job name
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=200gb                     # Job memory request
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/log/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2

# Default command line  args
# pretrain.py parameters
CFG_FILE=config/slidr_minkunet.yaml
RESUME_PATH=None
EXTRA_TAG=None
RANDOM_SEED=2022

# Additional parameters
SING_IMG=/raid/singularity/stslidr.sif
DATA_DIR=/raid/datasets/nuscenes
DATA_DIR_KITTI=/raid/datasets/semantic_kitti
SUPERPIXEL_DIR=/raid/datasets/nuscenes/superpixels/
WANDB_MODE='offline'
# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --gres=gpu:NUM_GPUS scripts/${0##*/} [-h]
pretrain.py parameters
[--cfg_file CFG_FILE]
[--resume_path RESUME_PATH]
[--extra_tag EXTRA_TAG]
[--random_seed RANDOM_SEED]

additional parameters
[--data_dir DATA_DIR]
[--superpixel_dir SUPERPIXEL_DIR]
[--sing_img SING_IMG]

pretrain.py parameters
--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--resume_path          RESUME_PATH        Resume path                         [default=$RESUME_PATH]
--extra_tag            EXTRA_TAG          Extra tag                           [default=$EXTRA_TAG]
--random_seed          RANDOM_SEED        Random seed                         [default=$RANDOM_SEED]

additional parameters
--data_dir             DATA_DIR           Zipped data directory               [default=$DATA_DIR]
--superpixel_dir       SUPERPIXEL_DIR     Superpixel data directory           [default=$SUPERPIXEL_DIR]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
"
}

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -r|--resume_path)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            RESUME_PATH=$2
            shift
        else
            die 'ERROR: "--resume_path" requires a non-empty option argument.'
        fi
        ;;
    -r|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -e|--random_seed)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            RANDOM_SEED=$2
            shift
        else
            die 'ERROR: "--random_seed" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -p|--superpixel_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SUPERPIXEL_DIR=$2
            shift
        else
            die 'ERROR: "--superpixel_dir" requires a non-empty option argument.'
        fi
        ;;
    -s|--sing_img)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SING_IMG=$2
            shift
        else
            die 'ERROR: "--sing_img" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

# Set extra tag to unique identifier if not specified
if [ $EXTRA_TAG == "None" ]
then
    EXTRA_TAG=$SLURM_JOB_NAME-$SLURM_JOB_ID
fi

BASE_CMD="SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_WANDB_MODE=$WANDB_MODE
singularity exec
--nv
--pwd /stslidr
--bind $PWD:/stslidr
--bind $DATA_DIR:/stslidr/datasets/nuscenes
--bind $DATA_DIR_KITTI:/stslidr/datasets/semantic_kitti
--bind $SUPERPIXEL_DIR:/stslidr/superpixels
$SING_IMG
"

PRETRAIN_CMD=$BASE_CMD

PRETRAIN_CMD+="python /stslidr/pretrain.py
--cfg_file $CFG_FILE
--extra_tag $EXTRA_TAG
--random_seed $RANDOM_SEED"

if [ $RESUME_PATH != "None" ]
then
    PRETRAIN_CMD+="--resume_path $RESUME_PATH
"
fi

echo "Running pretrain script"
echo "$PRETRAIN_CMD"
eval $PRETRAIN_CMD
echo "Done pretrain script"

echo "Running eval"
EVAL_CMD=$BASE_CMD
EVAL_CMD+="python /stslidr/post_pretrain.py
--cfg_file $CFG_FILE
--extra_tag $EXTRA_TAG
--random_seed $RANDOM_SEED"

FINETUNING_CMD=$EVAL_CMD
FINETUNING_CMD+="
--eval_type finetuning_nuscenes"

echo "Running finetuning"
echo "$FINETUNING_CMD"
eval $FINETUNING_CMD
echo "Done finetuning"

LINEAR_PROBE_CMD=$EVAL_CMD
LINEAR_PROBE_CMD+="
--eval_type linear_probe_nuscenes"

echo "Running linear probe"
echo "$LINEAR_PROBE_CMD"
eval $LINEAR_PROBE_CMD
echo "Done linear probe"

# semantic kitti
FINETUNING_CMD=$EVAL_CMD
FINETUNING_CMD+="
--eval_type finetuning_semkitti"

echo "Running finetuning"
echo "$FINETUNING_CMD"
eval $FINETUNING_CMD
echo "Done finetuning"


echo "Done eval"