#!/bin/bash

# List of commands to run
commands=(
    "bash scripts/adf.sh default $1 --test AD_hgt_2023-04-08-01:03"
    "bash scripts/adf.sh no_meta $1 --test AD_nm_2023-04-08-01:03"
    "bash scripts/adf.sh no_msg $1 --test AD_msg_2023-04-08-01:03"
    # Add more commands here
)

# Run each command in the background using a for loop
for command in "${commands[@]}"
do
    $command &
done

# Wait for all background processes to finish
wait

