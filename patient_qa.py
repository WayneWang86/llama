import subprocess
import pandas as pd

from datetime import datetime
import time

def main():
    # Define the command to be executed in the loop
    base_command = "torchrun --nproc_per_node 1 qa_pipeline.py " \
                   "--ckpt_dir llama-2-7b/ " \
                   "--tokenizer_path tokenizer.model " \
                   "--max_seq_len 128 --max_batch_size 4 "
    
    # Define array argument
    df = pd.read_csv("./data/forum_qa_pair.csv")
    questions = df["question"]

    left = 0
    right = 0
    n = len(questions)

    while left < n:
        print("left: ", left)
        if (n - left < 2):
            right += n - left
        else:
            right += 2
        print("right: ", right)
        print(datetime.now())

    
        # Add the additional argument to the base command
        command = base_command + f"--left {left} --right {right}"

        # Run the command using subprocess.run()
        subprocess.run(command, shell=True, check=True)
         

        print(datetime.now())
        print()
        left = right

        
#     res_texts = [res['generation']['content'] for res in results]
        
#     df = pd.DataFrame({"llama2-answer": res_texts})
    

#    for dialog in dialogs:
#        print(dialog)








# # Loop through the array and execute the command with each element
# for elem in my_array:
#     # Add the additional argument to the base command
#     command = base_command + f"--additional_arg {elem}"
    
#     # Run the command using subprocess.run()
#     subprocess.run(command, shell=True, check=True)
    
    
if __name__ == "__main__":
    main()