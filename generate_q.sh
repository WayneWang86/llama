#!/bin/bash

# Define the command to be executed in the loop
command="torchrun --nproc_per_node 1 process_prompt.py \
    --ckpt_dir /home/yifengw2/llama-2/llama/llama-2-7b-chat/ \
    --tokenizer_path /home/yifengw2/llama-2/llama/tokenizer.model \
    --max_seq_len 4096 --max_batch_size 4"


# Define array argument
# left=0
# n=12

# while [ "$left" -lt "$n" ]; do
#     echo "left: $left"
#     # if [ $((n - left)) -lt 3 ]; then
#     #     right=$((n - left))
#     # else
#     #     right=$((left + 3))
#     # fi
#     # echo "right: $right"
#     # echo $(date)
    
    # Update the environment variables
    # export LEFT=$left
    # export RIGHT=$right

#     # Run the command
#     $command

#     left=$left+1  # Update the 'left' value for the next iteration
# done

for i in {0..11}
do
    echo "Iteration number $i"
    # Update the environment variables
    export INDEX=$i
    # Run the command
    $command 
    
done