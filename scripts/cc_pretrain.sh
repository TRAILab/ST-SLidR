#!/bin/bash
#SBATCH --job-name=pretrain    # Job name
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=200gb                     # Job memory request
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/log/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-type=ALL

module load StdEnv/2020
module load apptainer

# Default command line  args
# pretrain.py parameters
CFG_FILE=config/pretrain/nuscenes/slidr/slidr_minkunet.yaml
RESUME_PATH=None
EXTRA_TAG=None
RANDOM_SEED=2022


# Additional parameters
STORAGE_ACCOUNT=rrg-swasland
SING_IMG=/home/$USER/projects/$STORAGE_ACCOUNT/singularity_images/stslidr.sif
DATA_DIR=/home/$USER/projects/$STORAGE_ACCOUNT/datasets/nuscenes
DATA_DIR_KITTI=/home/$USER/projects/$STORAGE_ACCOUNT/datasets/semantic_kitti
SUPERPIXEL_PREFIX=/home/$USER/projects/$STORAGE_ACCOUNT/datasets/nuscenes/superpixels
SUPERPIXEL_TYPE='slic'
WANDB_MODE='dryrun'

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
[--superpixel_prefix SUPERPIXEL_PREFIX]
[--superpixel_type SUPERPIXEL_TYPE]
[--sing_img SING_IMG]

pretrain.py parameters
--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--resume_path          RESUME_PATH        Resume path                         [default=$RESUME_PATH]
--extra_tag            EXTRA_TAG          Extra tag                           [default=$EXTRA_TAG]
--random_seed          RANDOM_SEED        Random seed                         [default=$RANDOM_SEED]

additional parameters
--data_dir             DATA_DIR           Zipped data directory               [default=$DATA_DIR]
--superpixel_prefix    SUPERPIXEL_PREFIX  Superpixel prefix data directory    [default=$SUPERPIXEL_PREFIX]
--superpixel_type      SUPERPIXEL_TYPE    Superpixel type                     [default=$SUPERPIXEL_TYPE]
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
    -p|--superpixel_prefix)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SUPERPIXEL_PREFIX=$2
            shift
        else
            die 'ERROR: "--superpixel_prefix" requires a non-empty option argument.'
        fi
        ;;
    -t|--superpixel_type)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SUPERPIXEL_TYPE=$2
            shift
        else
            die 'ERROR: "--superpixel_type" requires a non-empty option argument.'
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

# Add superpixel dir and type together
SUPERPIXEL_DIR=$SUPERPIXEL_PREFIX/$SUPERPIXEL_TYPE

echo "Running with the following arguments:
pretrain.py parameters:
CFG_FILE=$CFG_FILE
RESUME_PATH=$RESUME_PATH
EXTRA_TAG=$EXTRA_TAG
RANDOM_SEED=$RANDOM_SEED

Additional parameters
DATA_DIR=$DATA_DIR
SUPERPIXEL_DIR=$SUPERPIXEL_DIR
SING_IMG=$SING_IMG
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

# Extract Dataset
echo "Extracting data"
TMP_DATA_DIR=$SLURM_TMPDIR/data
for file in $DATA_DIR/*.zip; do
    echo "Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

echo "Extracting superpixels"
TMP_SUPERPIXEL_DIR=$SLURM_TMPDIR/superpixels
for file in $SUPERPIXEL_DIR/*.zip; do
    echo "Unzipping $file to $TMP_SUPERPIXEL_DIR"
    unzip -qq $file -d $TMP_SUPERPIXEL_DIR
done
echo "Done extracting data"

BASE_CMD="SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_WANDB_MODE=$WANDB_MODE
singularity exec
--nv
--pwd /stslidr
--bind $PWD:/stslidr
--bind $TMP_DATA_DIR:/stslidr/datasets/nuscenes
--bind $TMP_SUPERPIXEL_DIR:/stslidr/superpixels
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

# Nuscenes Evaluation
echo "Running nuscenes eval"
EVAL_CMD=$BASE_CMD
EVAL_CMD+="python /stslidr/post_pretrain.py
--cfg_file $CFG_FILE
--extra_tag $EXTRA_TAG
--random_seed $RANDOM_SEED"

FINETUNING_CMD=$EVAL_CMD
FINETUNING_CMD+="
--eval_type finetuning_nuscenes"

echo "Running nuscenes finetuning"
echo "$FINETUNING_CMD"
eval $FINETUNING_CMD
echo "Done nuscenes finetuning"

LINEAR_PROBE_CMD=$EVAL_CMD
LINEAR_PROBE_CMD+="
--eval_type linear_probe_nuscenes"

echo "Running nuscenes linear probe"
echo "$LINEAR_PROBE_CMD"
eval $LINEAR_PROBE_CMD
echo "Done linear probe"

echo "Done nuscenes eval"

# Extract Semantic Kitti 
echo "Remove nuscenes dataset"
ls -l $TMP_DATA_DIR
rm -r $TMP_DATA_DIR/*
ls -l $TMP_DATA_DIR
echo "Done removing nuscenes"

echo "unzipping semantic kitti"
for file in $DATA_DIR_KITTI/*.zip; do
    echo "Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

# Semantic Kitti Evaluation
BASE_CMD="SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_WANDB_MODE=$WANDB_MODE
singularity exec
--nv
--pwd /stslidr
--bind $PWD:/stslidr
--bind $TMP_DATA_DIR:/stslidr/datasets/semantic_kitti
$SING_IMG
"

echo "Running semantic kitti eval"
EVAL_CMD=$BASE_CMD
EVAL_CMD+="python /stslidr/post_pretrain.py
--cfg_file $CFG_FILE
--extra_tag $EXTRA_TAG
--random_seed $RANDOM_SEED"

FINETUNING_CMD=$EVAL_CMD
FINETUNING_CMD+="
--eval_type finetuning_semkitti"

echo "Running semantic kitti finetuning"
echo "$FINETUNING_CMD"
eval $FINETUNING_CMD
echo "Done semantic kitti finetuning"

echo "Done eval"