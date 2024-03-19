#Create a lora adapter, init it according to QLora paper.
import torch
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing
gc.collect()

LLM_cfgfile = "LLMbase_config.py"
# Import variables from the configuration file
exec(open(LLM_cfgfile).read())
input_cfgfile = "FedCS_LLMinput.py"
# Import variables from the configuration file
exec(open(input_cfgfile).read())
print("Read the config files")

import subprocess
packages_to_install = [
    "nltk",
    "rouge",
    "transformers==4.31.0",
    "datasets==2.13.0",
    "peft==0.4.0",
    "accelerate==0.21.0",
    "bitsandbytes==0.40.2",
    "trl==0.4.7",
    "safetensors>=0.3.1",
    "ipywidgets==7.7.1",
    "huggingface_hub",
    "python-dotenv",
    "scipy",
    "pandas"
]

# Install the packages using subprocess
#for package in packages_to_install:
#    subprocess.run(["pip", "install", package])
print("Installations completed")

output_dir = 'round0/server/'
adapter_dir = output_dir+'Ada/'

before_time = "NA"
after_time = "NA"

from datasets import load_dataset
from random import randrange
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaTokenizer, LlamaForCausalLM, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, get_peft_model_state_dict, AutoPeftModelForCausalLM, prepare_model_for_int8_training
from trl import SFTTrainer

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import os
import random
from datasets import Dataset
import csv
import json
from datasets import load_from_disk
from collections import OrderedDict
import datetime
import nltk
nltk.download('wordnet')
import sys
from io import StringIO
print("Imports completed")

import pandas as pd
import json
from huggingface_hub import login
from dotenv import load_dotenv
import os
credentials_filename = "../credentials.txt"
# Read the token from the file
token = None
with open(credentials_filename, "r") as file:
    for line in file:
        if "HF_HUB_TOKEN" in line:
            token = line.split("=")[1].strip()
            break
# Check if a token was found
if token is None:
    raise ValueError("HF_HUB_TOKEN not found in credentials.txt")
# Login to the Hugging Face Hub
login(token=token)
print("Logged into huggingface")

#print versions of Torch and cuda if available. 
def print_torch_cuda_info():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
print_torch_cuda_info()

import os
from datasets import Dataset
#Read the datafiles
import pandas as pd
import re
# BitsAndBytesConfig int-4 config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)
model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
model.config.pretraining_tp = 1
print("model loaded")

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# tokenizer.pad_token = 0
# tokenizer.padding_side = "left"
print("tokenizer defined")

import datetime
import torch
from peft import PeftModel
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
#Evaluate model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# TRN_PARAM = model.print_trainable_parameters()
captured_output = StringIO()
sys.stdout = captured_output
model.print_trainable_parameters()
sys.stdout = sys.__stdout__
TRN_PARAM = captured_output.getvalue()
print("Train parameters:", TRN_PARAM)

model.save_pretrained(adapter_dir)
print("Adapter saved")

# Empty VRAM
del model
import gc
gc.collect()
gc.collect()

from peft import AutoPeftModelForCausalLM
new_model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir, #args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
) #This is the adapters reloaded.

# Merge LoRA and base model
merged_model = new_model.merge_and_unload()
merged_model.save_pretrained(adapter_dir+"../merged_model/",safe_serialization=True)
# del new_model

#Read test dataset and add evaluation
test_dataset = load_from_disk(test_dataset_dir)
num_test_points = len(test_dataset)
print(num_test_points)
def generate_docstring(merged_model, sample, tokenizer):
    prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {"generate docstring for the below python function"}
    ### Input:
    {sample["function"]}
    ### Response:
    """
        # Tokenize the prompt and generate the docstring
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.5)
    # Extract and format the generated docstring
    generated_docstring = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    # Return the generated and ground truth docstrings
    return generated_docstring, sample['docstring']

# generated, ground_truth = generate_docstring(sample, tokenizer)
def evaluate_docstring(merged_model, test_dataset, tokenizer):
    # Initialize evaluation metrics
    rouge = Rouge()
    all_generated = []
    all_ground_truth = []

    # Loop over the entire test dataset to generate docstrings
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        generated, ground_truth = generate_docstring(merged_model, sample, tokenizer)
        all_generated.append(generated)
        all_ground_truth.append(ground_truth)
    #Health check of generated data:
    for idx, text in enumerate(all_generated):
        if not text.strip():  # If text is empty or just whitespace
            print(f"Empty string found at index {idx}")
            all_generated[idx] = "NO_OUTPUT"
    for idx, text in enumerate(all_ground_truth):
        if not text.strip():  # If text is empty or just whitespace
            print(f"Empty reference found at index {idx}")
            all_ground_truth[idx] = "NO_INPUT"
            all_generated[idx] = "SO_NO_OUTPUT"

    # Evaluate Corpus-BLEU
    bleu_score = corpus_bleu([[gt.split()] for gt in all_ground_truth], [g.split() for g in all_generated])

    # Evaluate METEOR
    meteor_scores = [meteor_score([gt.split()], g.split()) for gt, g in zip(all_ground_truth, all_generated)]
    average_meteor = sum(meteor_scores) / len(meteor_scores)

    # Evaluate ROUGE
    rouge_scores = rouge.get_scores(all_generated, all_ground_truth, avg=True)

    print(f'Corpus-BLEU: {bleu_score}')
    print(f'Average METEOR: {average_meteor}')
    print(f'ROUGE: {rouge_scores}')

    # Return the results
    evaluation_results = {
        'BLEU': bleu_score,
        'METEOR': average_meteor,
        'ROUGE': rouge_scores
    }

    return evaluation_results

#metric_results = evaluate_docstring(merged_model, test_dataset, tokenizer)
model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
metric_results = evaluate_docstring(model, test_dataset, tokenizer)

# Define the file path and column headers
file_path = history_file_path
column_headers = ["Data", "TrainDataSize", "ModelType", "Round","BaseModelName", "TrainableParams", "TargetModules", "RankLoRAModule", "StartTime", "EndTime", "NumberEpochs", "EvalDataSize", "C-BLEU", "METEOR", "ROUGE-L"]

# Check if the file already exists
if not os.path.exists(file_path):
    # Create the CSV file and write the headers
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_headers)
    print(f"File '{file_path}' created successfully.")
else:
    print(f"File '{file_path}' already exists.")
    
with open(file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    this_row=["NA", "NA", "FedAvg", "0", model_id, TRN_PARAM, LORA_TARGET_MODULES, lora_r ,before_time, after_time, num_train_epochs, len(test_dataset), metric_results['BLEU'],  metric_results['METEOR'],metric_results['ROUGE']]
    writer.writerow(this_row)
print("History file updated with round0 eval metrics")