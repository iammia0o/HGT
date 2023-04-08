#!/bin/bash
arg1=$1
arg2=${2:-20}
arg3=${3:-""}
arg4=${4:-""}
home_dir=/home/mia/repo/HGT

# Set base command
base_command="python3 ${home_dir}/train_paper_field.py --data_dir ${home_dir}/data \
    --model_dir ${home_dir}/save_model --conv_name hgt --domain '_CS' --level 1 --cuda 1 --n_epoch $arg2"

# Check for additional options based on the value of cmd
if [[ "$1" == "no_meta" ]]; then
    # Add options for no_meta
    base_command="$base_command --model_name nm --no_meta_rela"
elif [[ "$1" == "no_msg" ]]; then
    # Add options for no_msg
    base_command="$base_command --model_name msg --no_msg_rela"
fi

if [[ "$arg3" == "--test" ]]; then 
    if [ -z "$arg4" ]; then
        echo "need to set saved model path"
    fi
    base_command="$base_command --test --saved_model $arg4"
fi

# Run the command
echo "Running command: $base_command"
eval "$base_command"
