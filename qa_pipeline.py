# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, List

import fire
import pandas as pd
import ast
import os

from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    left = os.environ.get('LEFT')
    right = os.environ.get('RIGHT')
    
    df = pd.read_csv("./data/question_categories.csv")
    questions = df["questions"][int(left):int(right)]
    # questions = df["questions"][159:160]
    dialogs= [[{"role": "user", "content": q}] for q in questions]
    
    # print(dialogs)
    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #         print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
    #         print("\n==================================\n")
    
    res_texts = [res['generation']['content'] for res in results]
    df = pd.DataFrame({"question": questions, "llama2-answer": res_texts})
    
    # Name of the existing file where you want to add the array elements
    file_path = './data/llama2_response.csv'
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df])
    
    df.to_csv(file_path, index=False)   

if __name__ == "__main__":
    fire.Fire(main)
