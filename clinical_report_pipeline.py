# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, List

import fire
import pandas as pd
import ast
import os

from llama import Llama, Dialog
from tqdm import tqdm

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.3,
    top_p: float = 0.5,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    
    index = os.environ.get('INDEX')
    # index = 8
    # right = os.environ.get('RIGHT')
    
    data = pd.read_csv("./data/sample_noteevent.csv")
    
    # print("dimensions of df: ", data.shape)
    patients=[1028, 1029,1050,1063]
    sample = data[data["SUBJECT_ID"].isin(patients)].sort_values("SUBJECT_ID")
    # sample_notes = ["#Clinical Notes: \n\n" + t if len(t) < 10000 else "#Clinical Notes: \n\n" + t[:10000] for t in sample["TEXT"]]
    sample_notes = ["#Clinical Notes: \n\n" + t for t in sample["TEXT"]]    
    # print(f"sample notes length: {len(sample_notes)}")
    # print(f"sample notes: {sample_notes[0]}")
    
#     instruction = """
# You are an AI clinical assistant with the task to summarize and reorganize clinical notes for the physicians. 

# Please summarize the patient's related medical history for me, which includes  Medical and family history, recent events (e.g., surgeries), reason for admission.

# Give me your response following this exact format:
# “Patient Info”: [YOUR RESPONSE]
#     """
    
# # History & Condition: Medical and family history, recent events (e.g., surgeries), reason for admission.
# # Medications & Treatments: Pre-admission medications, changes during stay, new prescriptions at discharge, and treatments like surgery.
# # Test Results & Findings: Lab tests, imaging, physical exams from the stay, highlighting key values and observations.
# # Discharge & Follow-Up: Condition at discharge, home care instructions, medications, follow-up appointments, other recommendations.

# # “History & Condition”: …
# # “Medications & Treatments”: …
# # “Test Results & Findings”: …
# # “Discharge & Follow-Up”: …

    instruction = """
    You are an AI clinical assistant with the task to summarize and reorganize clinical notes for the physicians. Summarize the clinical notes into five sections without adding extra information:

Patient Info: Admission/discharge dates, birth date, sex, treatment location.
History & Condition: Medical and family history, recent events (e.g., surgeries), reason for admission.
Medications & Treatments: Pre-admission medications, changes during stay, new prescriptions at discharge, and treatments like surgery.
Test Results & Findings: Lab tests, imaging, physical exams from the stay, highlighting key values and observations.
Discharge & Follow-Up: Condition at discharge, home care instructions, medications, follow-up appointments, other recommendations.
Format the output with each section titled and followed by relevant information, like the example:

“Patient Demographics and Admission Info”: … 
 “Medical History and Current Condition”: …
 “Medications and Treatments”: …
 “Test Results and Clinical Findings”: …
 “Discharge Plan and Follow-Up Care”: …

    """
    
    results = []
    
    # for i in tqdm(range(len(sample_notes))):
    dialogs = [[{"role": "system", "content": instruction}, {"role": "user", "content": sample_notes[int(index)]}]]

    # dialogs = [[{"role": "system", "content": instruction}, {"role": "user", "content": x}] for x in samples]

    # # print(dialogs)

    result = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    results.append(result[0]['generation']['content'])
    
    print(result[0]['generation']['content'])
# # for dialog, result in zip(dialogs, results):
# #     for msg in dialog:
# #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
# #         print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
# #         print("\n==================================\n")

    # res_texts = [res['generation']['content'] for res in results]
    df = pd.DataFrame({"patient_id": list(sample["SUBJECT_ID"])[int(index)], "note_event": sample_notes[int(index)], "summaries": results})
    
    # Name of the existing file where you want to add the array elements
    file_path = './data/summaries.csv'
    # file_path = './data/patient_info.csv'
    # file_path = './data/history.csv'
    # file_path = './data/med.csv'
    # file_path = './data/test.csv'
    # file_path = './data/discharge.csv'
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df])
    
    df.to_csv(file_path, index=False)   

if __name__ == "__main__":
    fire.Fire(main)
