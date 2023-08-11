#!/bin/bash

# Define the command to be executed in the loop
command="torchrun --nproc_per_node 1 qa_pipeline.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 4"


# Define array argument
left=0
right=3
n=1524

while [ "$left" -lt "$n" ]; do
    echo "left: $left"
    if [ $((n - left)) -lt 3 ]; then
        right=$((n - left))
    else
        right=$((left + 3))
    fi
    echo "right: $right"
    echo $(date)
    
    # Update the environment variables
    export LEFT=$left
    export RIGHT=$right

    # Run the command
    $command

    left=$right  # Update the 'left' value for the next iteration
done
