# For Round in range(1, 20): //Later can repeat for 20 more!!
# 	For loop for all clients:
# 		server to share AdaAvg to client
# 		Train client-i with DS-i. Save the trained adapter weights(Send to server). 
# 		Clean GPU.
# 	Server to create and save AdaAVg. Update validation for round++. Clean GPU.
import torch
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing
gc.collect()

#Create a lora adapter, init it according to QLora paper.
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
# for package in packages_to_install:
#     subprocess.run(["pip", "install", package])
print("Installations completed")

# # Set PYTORCH_CUDA_ALLOC_CONF environment variable
# pytorch_cuda_alloc_conf = "0:11500"
# # Set PYTORCH_CUDA_ALLOC_CONF environment variable
# subprocess.run(["export", f"PYTORCH_CUDA_ALLOC_CONF={pytorch_cuda_alloc_conf}"])

from datasets import load_dataset
from random import randrange
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaForCausalLM, Trainer
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
# Define the filename for the credentials file
credentials_filename = "../credentials.txt"
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

#print versions of Torch and cuda if available. 
def print_torch_cuda_info():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
print_torch_cuda_info()

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

def eval_FedAvgServer(merged_model, tokenizer, Fed_round, TRN_PARAM, before_time, after_time):
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
        this_row=["code_docstring_corpus_data",num_train_rows , "FedAvg", Fed_round, model_id, TRN_PARAM, LORA_TARGET_MODULES, lora_r ,before_time, after_time, num_train_epochs, len(test_dataset), metric_results['BLEU'],  metric_results['METEOR'],metric_results['ROUGE']]
        writer.writerow(this_row)
    return "Updated"

# Server side: avg the client adapters
"""
FedAvg the adapters
Parameters: num_clients is the number of clients, client_names are list of client names, 
adapter_locations are list of locations, trainrows_details are the list of  #trainrows used for each adapter tuning
"""
def read_adapter(adapter_dir): 
    adapter_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir, #args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    ) 
    return adapter_model
def get_lora_states(adapter):
    my_state_dict = adapter.state_dict()
    lora_return = {}
    for k in my_state_dict:
        if 'lora_' in k:
            lora_return[k] = my_state_dict[k]
            bias_name = k.split('lora_')[0]+'bias'
            if bias_name in my_state_dict:
                lora_return[bias_name] = my_state_dict[bias_name]
    return lora_return
def FedAvg(num_clients, adapter_locations, trainrows_details):
    if num_clients <1:
        print("error: FedAvg can only be applied on 1+ clients")
        return None
    elif num_clients ==1:
        return read_adapter(adapter_locations[0])
    
    #Normalize the training rows: feature scaling / min-max scaling to range(0,1)
    totalrows = sum(trainrows_details)
    normalized_trainrows = [value / totalrows for value in trainrows_details]
    
    lora_adapters = []
    for i in range(0, num_clients): #1..num_clients
        #Read num_trainrows, adapter_locations
        this_adapter = read_adapter(adapter_locations[i]) #read the ith adapter
        this_lora = get_lora_states(this_adapter)
        del this_adapter
        lora_adapters.append(this_lora)
    
    #Create a avg_adapter model by weighted averaging all weights in given models in models list. normalized_trainrows list has the respectiev weight to be multiple with for each model.
    # Initialize the average adapter
    avg_adapter_state_dict = OrderedDict()
    
    # LoRA weights are averaged here
    for key in lora_adapters[0].keys():
        avg_adapter_state_dict[key] = sum(
            lora_adapters[i][key] * normalized_trainrows[i]
            for i in range(num_clients)
        )
    del lora_adapters
    avg_loraadapter = read_adapter(adapter_locations[0]) #Initialization  
    # Copy all non-lora rows
    for key in avg_loraadapter.state_dict().keys():
        if key not in avg_adapter_state_dict.keys():
            avg_adapter_state_dict[key] = avg_loraadapter.state_dict()[key]
#     del avg_loraadapter

    # Initialize a new adapter model with the averaged weights
#     avg_loraadapter = AutoPeftModelForCausalLM.from_pretrained(adapter_locations[0])  # Initialize from the first adapter
    avg_loraadapter.load_state_dict(avg_adapter_state_dict)
    return avg_loraadapter

def FedAvg_withRefLoad(num_clients, adapter_locations, dataset_ratio_split4clients, prevmodel_dir):
#     model = merge_lora2base(prevmodel_dir)
#     merged_model_dir = prevmodel_dir+"../merged_model"
#     model.save_pretrained(merged_model_dir,safe_serialization=True)
#     del model
    avg_loraadapter = FedAvg(num_clients, adapter_locations, dataset_ratio_split4clients)
#     delete_directory(merged_model_dir)
    return avg_loraadapter

#Config to finetune / merge the models

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer.pad_token = 0
# tokenizer.padding_side = "left"
print("tokenizer defined")
# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES
)
#Config to finetune the model
import shutil
def delete_directory(directory):
    # Define the directory you want to delete
    merged_model_dir = directory
    # Delete the contents of the directory
    try:
        shutil.rmtree(merged_model_dir)
#         print(f"Contents of {merged_model_dir} deleted successfully.")
    except FileNotFoundError:
        print(f"The del_directory {merged_model_dir} does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the contents of {merged_model_dir}: {str(e)}")

def get_model_tobe_trained(prevmodel_dir):
    merged_model_dir = prevmodel_dir+"../merged_model"
#     model.save_pretrained(merged_model_dir,safe_serialization=True)
#     del model
    model = AutoModelForCausalLM.from_pretrained(merged_model_dir, use_cache=False, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
#     delete_directory(merged_model_dir)
    model.config.pretraining_tp = 1
    print("model loaded")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    return model

def train_save_model(model, adapter_dir, dataset):
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
    # Set the instruction format for iamtarun/python_code_instructions_18k_alpaca
    def format_instruction(sample):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {"generate docstring for the below python function"}
    ### Input:
    {sample["function"]}
    ### Response:
    {sample["docstring"]}"""
    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        formatting_func=format_instruction,
        args=args,
        data_collator=data_collator
    )
    print("model trainer declared")

    before_time = datetime.datetime.now()
    before_time_str = before_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print("before time_str", before_time_str)
    trainer.train() # there will not be a progress bar since tqdm is disabled
    after_time = datetime.datetime.now()
    # Format datetime objects into strings
    after_time_str = after_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print("after time_str", after_time_str)
    # save model in local
    # trainer.save_model()
    model.save_pretrained(adapter_dir)
    print("Model trained and saved")
    del trainer
    del model
    return before_time_str, after_time_str

#Config to merge the model

def merge_lora2base(adapter_dir):
    new_model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, low_cpu_mem_usage=True, 
    return_dict=True, torch_dtype=torch.float16, device_map=device_map,) #This is the adapters reloaded.
    # Merge LoRA and base model
    merged_model = new_model.merge_and_unload()
    return merged_model
def merge_lora2base_withRedLoad(adapter_dir, prevmodel_dir):
#     model = merge_lora2base(prevmodel_dir)
#     merged_model_dir = prevmodel_dir+"/merged_model"
#     model.save_pretrained(merged_model_dir,safe_serialization=True)
#     del model
    merged_model = merge_lora2base(adapter_dir)
#     delete_directory(merged_model_dir)
    return merged_model

#Config to Fed run
from datasets import load_from_disk
train_dataset = load_from_disk(train_dataset_dir)
num_train_points = len(train_dataset)
print("Number of trin rows", num_train_points)
del train_dataset
#Calculate dataset split
total_sum = sum(dataset_ratio_split4clients)
dataset_ratio_split4clients = [x / total_sum*num_train_points for x in dataset_ratio_split4clients]
# convert float to int by floor value
dataset_ratio_split4clients = [int(x) if isinstance(x, (int, float)) else x for x in dataset_ratio_split4clients]
# If all 3 doesnt add upto num_train_points, then increase last one.
dataset_ratio_split4clients[num_clients-1] += num_train_points - sum(dataset_ratio_split4clients)
dataset_ratio_split4clients
print("Train split data generated: ", dataset_ratio_split4clients)

#After server run, updates
def update_basemodel_inAdaCfg(adapter_dir):
    config_file_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
    config_data["base_model_name_or_path"] = model_id
    with open(config_file_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

for this_round in range(1,num_rounds+1):
    print("beginning Round: ",this_round)
    ##Each clients job
    adapter_locations = []
    before_times = []
    after_times = []
    #Read server lora model from this_round-1
    lastround_server_dir = 'round'+str(this_round -1)+'/server/'+'Ada/'
    for this_client in range(0,num_clients):
        print("      ",this_round,"Running for client_",this_client)
        output_dir = 'round'+str(this_round)+'/client_'+str(this_client)+'/'
        adapter_dir = output_dir+'Ada/'
        adapter_locations.append(adapter_dir)
        model = get_model_tobe_trained(lastround_server_dir)
        
        # TRN_PARAM = model.print_trainable_parameters()
        captured_output = StringIO()
        sys.stdout = captured_output
        model.print_trainable_parameters()
        sys.stdout = sys.__stdout__
        TRN_PARAM = captured_output.getvalue()
        print("Train parameters:", TRN_PARAM)

        this_train_dataset_dir = "client_datasets/client_"+str(this_client)+'/'
        train_dataset = load_from_disk(this_train_dataset_dir)
        before_time, after_time = train_save_model(model, adapter_dir, train_dataset)
        before_times.append(before_time)
        after_times.append(after_time)
        del model
        print("      Model saved for client_",this_client)
    
    ##Server's job
    #Calculate AdaAvg
    avg_ada = FedAvg_withRefLoad(num_clients, adapter_locations, dataset_ratio_split4clients, lastround_server_dir)

    #Save AdaAvg
    output_dir = 'round'+str(this_round)+'/server/'
    adapter_dir = output_dir+'Ada/'
    avg_ada.save_pretrained(adapter_dir)
    print("Model Ada saved in server dir ")
    del avg_ada
    #Validation
    merged_model = merge_lora2base_withRedLoad(adapter_dir, lastround_server_dir)
    print("Merged Model loaded")
    if this_round >= 2:
        delete_directory('round'+str(this_round-2)+'/server/merged_model/')
        print(this_round-2,"server merged model deleted")
    merged_model.save_pretrained(adapter_dir+"../merged_model/",safe_serialization=True) #TBW
    print("Merged Model saved")
    status = eval_FedAvgServer(merged_model, tokenizer, this_round, TRN_PARAM, before_times, after_times)
    del merged_model
    if status!="Updated":
        print("ERROR: Updating validation history file failed after round",this_round, ", please check.")
    else:
        print("Validation history file updated after Fed round: ",this_round)
