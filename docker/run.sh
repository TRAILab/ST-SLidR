#!/bin/bash

CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR)
NUSCENES_RAW=None
BASH_HISTORY_FILE=~/.docker_stslidr_bash_history
ZSH_HISTORY_FILE=~/.docker_stslidr_zsh_history

# Usage info
show_help() {
echo "
Usage: ./run.sh [-h]
[--nuscenes]
[--target TARGET]
[--bash_history_file BASH_HISTORY_FILE]
[--zsh_history_file ZSH_HISTORY_FILE]
--bash_history_file BASH_HISTORY_FILE Bash history file                [default=$BASH_HISTORY_FILE]
--zsh_history_file   ZSH_HISTORY_FILE Zsh history file                 [default=$ZSH_HISTORY_FILE]
"
}

while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -w|--nuscenes)       # Takes an option argument; ensure it has been specified.
        NUSCENES_RAW=$(readlink -f ../datasets/nuscenes):/stslidr/datasets/nuscenes
        ;;
    -b|--bash_history_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BASH_HISTORY_FILE=$2
            shift
        else
            die 'ERROR: "--bash_history_file" requires a non-empty option argument.'
        fi
        ;;
    -z|--zsh_history_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            ZSH_HISTORY_FILE=$2
            shift
        else
            die 'ERROR: "--zsh_history_file" requires a non-empty option argument.'
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

# Ensure history files exist
touch $BASH_HISTORY_FILE
touch $ZSH_HISTORY_FILE
echo ${NUSCENES_RAW}
docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$XAUTHORITY:/root/.Xauthority:rw" \
        --volume $BASH_HISTORY_FILE:/home/ddet/.bash_history \
        --volume $ZSH_HISTORY_FILE:/home/ddet/.zsh_history \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="WANDB_API_KEY=$WANDB_API_KEY" \
        --hostname="inside-DOCKER" \
        --name="stslidr" \
        --volume $PROJ_DIR:/stslidr \
        --volume ${NUSCENES_RAW} \
        --rm \
        stslidr bash