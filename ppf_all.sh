#!/bin/bash

# List of commands to run
commands=(
    "bash scripts/ppf2.sh default $1 --test PF_hgt_2023-04-08-01:03"
    "bash scripts/ppf2.sh no_meta $1 --test PF_nm_2023-04-08-01:03"
    "bash scripts/ppf2.sh no_msg $1 --test PF_msg_2023-04-08-01:03"
    # Add more commands here
)

# Run each command in the background using a for loop
for command in "${commands[@]}"
do
    $command &
done

# Wait for all background processes to finish
wait

