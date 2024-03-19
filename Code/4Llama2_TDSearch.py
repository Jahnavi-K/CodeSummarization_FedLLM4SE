#cosine scheduler coz rouge-l and meteor is better in cosine than cnst. The TRN param is coppied as its populating in the csv.
#Create a lora adapter, init it according to QLora paper. eval it acc to Qlora paper. all compile both for train and inference is removed.
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

output_dir = "NonFed/"
test_dataset_dir = "../code_docstring_corpus_data_test/"
all_train_dataset_dir = "../code_docstring_corpus_data_alltrain/"
train_dataset_dir = "../code_docstring_corpus_data_train/"
eval_dataset_dir = "../code_docstring_corpus_data_eval/"
history_file_path = "../Laama2_RunHistory_LoRA_TDSearch.csv"
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

#Create Train data
import os
from datasets import Dataset
#Read the datafiles
import pandas as pd
import re
#import sklearn
#import scikit-learn.model_selection import train_test_split
# Define the folder path
folder_path = "../code_docstring_corpus_data"

# Function to create a dataset for a given file type (train, test, valid)
def create_dataset(file_type):
    data = {"function": [], "docstring": []}
    # Define the paths to the declbodies and descriptions files
    declbodies_file_path = os.path.join(folder_path, f"data_ps.declbodies.{file_type}")
    descriptions_file_path = os.path.join(folder_path, f"data_ps.descriptions.{file_type}")
    # Read and process the files
    with open(declbodies_file_path, "r", encoding="latin-1") as declbodies_file, open(descriptions_file_path, "r", encoding="latin-1") as descriptions_file:
        declbodies_lines = declbodies_file.readlines()
        descriptions_lines = descriptions_file.readlines()

        # Ensure that the number of lines in both files match
        if len(declbodies_lines) != len(descriptions_lines):
            raise ValueError(f"Number of lines in {file_type} declbodies and descriptions files do not match.")

        for declbody, description in zip(declbodies_lines, descriptions_lines):
            # Preprocess declbody to replace "DCNL" with newline and "DCSP" with tab space
            declbody = declbody.replace("DCNL", "\n")
            declbody = declbody.replace("DCSP", "\t") if declbody.startswith("\n") else declbody.replace("DCSP", " ")
            description = description.replace("DCNL", "\n")
            description = description.replace("DCSP", "\t") if description.startswith("\n") else description.replace("DCSP", " ")
            declbody = declbody.strip()
            description = description.strip()
            
            data["function"].append(declbody)
            data["docstring"].append(description)

    # Create and return the Dataset object
    return data

def create_TD_for_training(this_TDsize):
    train_dataset = create_dataset("train")
    train_data = Dataset.from_dict(train_dataset)
    train_data.save_to_disk(all_train_dataset_dir)
    del train_data, train_dataset
    train_data = load_from_disk(all_train_dataset_dir)
    # Get the number of rows in the dataset
    num_rows = len(train_data)
    # Generate a list of all indices
    all_indices = list(range(num_rows))
    # Generate a random index for the first 10,000 rows for the training dataset
    train_indices = random.sample(all_indices, this_TDsize)
    train_dataset = train_data.select(train_indices)
    final_num_rows = len(train_dataset)
    train_dataset.save_to_disk(train_dataset_dir)
    del train_dataset, train_data
    print("Train data generated with #rows:", final_num_rows)

    test_dataset = create_dataset("test")
    test_data = Dataset.from_dict(test_dataset)
    test_data.save_to_disk(test_dataset_dir)
    del test_data, test_dataset
    print("Test data generated")
    return final_num_rows

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = 0
tokenizer.padding_side = "left"
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
train_dataset = load_from_disk(train_dataset_dir)

# #Cases description
# weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
# adapter_dir = output_dir+'Ada_W'+weight_matrices+'_r'+str(lora_r)+'/'
def format_instruction(sample):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{"generate docstring for the below python function"}
### Input:
{sample["function"]}
### Response:
{sample["docstring"]}"""

def train_this_model(LORA_TARGET_MODULES, lora_r):
    # LoRA config based on QLoRA paper
    print("Model config: r=", lora_r, " Target modules", LORA_TARGET_MODULES )
    peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES
    )

    # Load the pretrained model
    model = LlamaForCausalLM.from_pretrained(model_id, use_cache = False, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
    model.config.pretraining_tp = 1
    print("model loaded")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)

    # TRN_PARAM = str(model.print_trainable_parameters())
    # Create a StringIO object to capture the output
    captured_output = StringIO()
    # Redirect stdout to the StringIO object
    sys.stdout = captured_output
    # Call the function that prints the information
    model.print_trainable_parameters()
    # Restore the original stdout
    sys.stdout = sys.__stdout__
    # Get the captured output as a string
    TRN_PARAM = captured_output.getvalue()
    print("Train parameters:", TRN_PARAM)
    print("model prepped with peft config")

    # Define the training arguments
    args = TrainingArguments(
    output_dir=output_dir, num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size, #per_device_train_batch_size, # 6 if use_flash_attention else 4, trying 2 ton mng mem
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing, optim=optim, #save_steps=save_steps,
    logging_steps=logging_steps, save_strategy="no", #"epoch", 
    learning_rate=learning_rate, weight_decay=weight_decay, fp16=fp16, bf16=bf16, max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio, #max_steps=max_steps,
    group_by_length=group_by_length, lr_scheduler_type=lr_scheduler_type,
    disable_tqdm=disable_tqdm, seed=42)

    data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    print("data collector is read")
    train_dataset = load_from_disk(train_dataset_dir)
    # trainer = Trainer(
    #     model=model, train_dataset=train_chosen_data, tokenizer=tokenizer, args=args, 
    #     eval_dataset=test_dataset, data_collator=data_collator)
    trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    # eval_dataset=val_dataset,
    args=args,
    data_collator=data_collator,
    packing=packing,
    formatting_func=format_instruction,
    )  
    print("model trainer declared")
    # model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    # model = torch.compile(model) #with compile, we get error: huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..To disable this warning, you can either:- Avoid using `tokenizers` before the fork if possible- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    # print("model compiled")

    before_time = datetime.datetime.now()
    trainer.train() # there will not be a progress bar since tqdm is disabled
    after_time = datetime.datetime.now()
    # Format datetime objects into strings
    before_time_str = before_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print("before time_str", before_time_str)
    after_time_str = after_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print("after time_str", after_time_str)
    # save model in local
    # trainer.save_model()
    model.save_pretrained(adapter_dir)
    print("Model trained and saved")
    del trainer
    del model
    return TRN_PARAM, before_time_str, after_time_str

# def create_prompt(data_point: str):
#     # Define the prompt using the input function
#     prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {"generate docstring for the below python function"}
# ### Input:
# {data_point["function"]}
# ### Response:
# """
#     return prompt

# def generate_response(prompt: str, model: PeftModel, tokenizer) -> GreedySearchDecoderOnlyOutput:
#     encoding = tokenizer(prompt, return_tensors="pt")
#     input_ids = encoding["input_ids"].to(DEVICE)
 
#     generation_config = GenerationConfig(
#         temperature=0.5, #0.1,
#         top_p=0.9, #0.75,
#         #repetition_penalty=1.1,
#     )
#     with torch.inference_mode():
#         return model.generate(
#             input_ids=input_ids,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             max_new_tokens=256,
#         )
    
# def format_response(response: GreedySearchDecoderOnlyOutput, tokenizer):
#     decoded_output = tokenizer.decode(response.sequences[0])
#     response = decoded_output.split("### Response:")[1].strip()
#     return "\n".join(textwrap.wrap(response))

# def ask_llama2(prompt, model, tokenizer):
#     prompt = create_prompt(prompt)
#     response = generate_response(prompt, model, tokenizer)
#     return format_response(response, tokenizer)

def generate_docstring(merged_model, sample, tokenizer):
    # generated_docstring = ask_llama2(sample , merged_model, tokenizer)
    # # Return the generated and ground truth docstrings
    # return generated_docstring, sample['docstring']
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

def eval_FedAvgServer(merged_model, tokenizer, Fed_round, TRN_PARAM, before_time, after_time, train_num_rows):
    test_dataset = load_from_disk(test_dataset_dir)
    metric_results = evaluate_docstring(merged_model, test_dataset, tokenizer)
    # Define the file path and column headers
    file_path = history_file_path
    column_headers = ["Data", "TrainDataSize", "ModelType", "Round","BaseModelName", "TrainableParams", "TargetModules", "RankLoRAModule", "StartTime", "EndTime", "NumberEpochs", "EvalDataSize", "C-BLEU", "METEOR", "ROUGE-L"]

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Create the CSV file and write the headers
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_headers)
#         print(f"File '{file_path}' created successfully.")
#     else:
#         print(f"File '{file_path}' already exists.")

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        this_row=["code_docstring_corpus_data", train_num_rows, "Central", "NA", model_id, TRN_PARAM, LORA_TARGET_MODULES, lora_r ,before_time, after_time, num_train_epochs, len(test_dataset), metric_results['BLEU'],  metric_results['METEOR'],metric_results['ROUGE']]
        writer.writerow(this_row)
    return "Updated"

def merge_lora2base(adapter_dir):
    new_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, low_cpu_mem_usage=True,
    return_dict=True, torch_dtype=torch.float16, device_map=device_map,) #This is the adapters reloaded.
    # Merge LoRA and base model
    merged_model = new_model.merge_and_unload()
    return merged_model

def this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, train_num_rows):
    this_round = "NA"
    # before_times = before_time_str
    # after_times = after_time_str
    TRN_PARAM, before_times, after_times = train_this_model(LORA_TARGET_MODULES, lora_r)
    merged_model = merge_lora2base(adapter_dir)
    #Set model to evaluation mode
    # merged_model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # merged_model.config.bos_token_id = 1
    # merged_model.config.eos_token_id = 2
    # merged_model = merged_model.eval()
    # merged_model = torch.compile(merged_model)
    # print("Read the merged model, abt to evaluate now")
    status = eval_FedAvgServer(merged_model, tokenizer, this_round, TRN_PARAM, before_times, after_times, train_num_rows)
    del merged_model
    if status!="Updated":
        print("ERROR: Updating validation history file failed after Central model", "please check.")
    else:
        print("Validation history file updated after Central model for r=", lora_r,"and W=", weight_matrices )

#RQ3: Identify Training Data size 
LORA_TARGET_MODULES = ["q_proj","k_proj"]
lora_r = 8
TD_list = [100, 1000, 10000, 25000, 50000, 109108]
for this_TD in TD_list:
    print("Creating dataset:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    train_num_rows = create_TD_for_training(this_TD)
    print("\n\n----------------------------------------------------------------------")
    print("Beginning experiment for W=", LORA_TARGET_MODULES, "with r=", lora_r, "for DS=", this_TD)
    print("EXP Start time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    weight_matrices = ''.join([module[0] for module in LORA_TARGET_MODULES])
    adapter_dir = output_dir+'/RQ3/TD'+str(this_TD)+'Ada_W'+weight_matrices+'_r'+str(lora_r)+'/'
    this_r_W_experiment(adapter_dir, weight_matrices, LORA_TARGET_MODULES, lora_r, train_num_rows)
    print("SEeeeeee!! experiment completed for W=", LORA_TARGET_MODULES, "with r=", lora_r, "for DS=", this_TD)
    print("EXP End time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing
gc.collect()