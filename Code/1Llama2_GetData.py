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

def create_dataset_train(file_type, train_type):
    data = {"function": [], "docstring": []}
    repo_counts = {}
    ref_df = pd.read_csv('df_selectNsplit.csv')
    #Create for Non-Fed
    # del repository_list
    if (train_type == "all"):
        repository_list = ref_df['Repository'].tolist()
    #Create for Fed client
    else:
        filtered_df = ref_df[ref_df['Client'] == train_type]
        repository_list = filtered_df['Repository'].tolist()
    
    print(f"Loaded {len(repository_list)} repositories for client type {train_type}")

    # Print out a sample of repository names for debugging
    # print("Sample repositories:", repository_list[:5])

    # Define the paths to the declbodies and descriptions files
    declbodies_file_path = os.path.join(folder_path, f"data_ps.declbodies.{file_type}")
    descriptions_file_path = os.path.join(folder_path, f"data_ps.descriptions.{file_type}")
    metadata_file_path = os.path.join(folder_path, f"data_ps.metadata.{file_type}")
    total_functions =0
    # Read and process the files
    with open(declbodies_file_path, "r", encoding="latin-1") as declbodies_file, open(descriptions_file_path, "r", encoding="latin-1") as descriptions_file, open(metadata_file_path, "r", encoding="latin-1") as metadata_file:
        declbodies_lines = declbodies_file.readlines()
        descriptions_lines = descriptions_file.readlines()
        metadata_lines = metadata_file.readlines()

        # Ensure that the number of lines in both files match
        if (len(declbodies_lines) != len(descriptions_lines)) or (len(metadata_lines) != len(descriptions_lines)):
            raise ValueError(f"Number of lines in {file_type} declbodies, descriptions and metadat files do not match.")
        #Read the selected repositories
        for declbody, description, metadata in zip(declbodies_lines, descriptions_lines, metadata_lines):
            # Preprocess declbody to replace "DCNL" with newline and "DCSP" with tab space
            meta_parts = metadata.split('/')
            repository_name = '/'.join(meta_parts[1:3]) + '/'

            if (repository_name in repository_list):
                total_functions +=1
                declbody = declbody.replace("DCNL", "\n")
                declbody = declbody.replace("DCSP", "\t") if declbody.startswith("\n") else declbody.replace("DCSP", " ")
                description = description.replace("DCNL", "\n")
                description = description.replace("DCSP", "\t") if description.startswith("\n") else description.replace("DCSP", " ")
                declbody = declbody.strip()
                description = description.strip()
                
                data["function"].append(declbody)
                data["docstring"].append(description)

                repo_counts[repository_name] = repo_counts.get(repository_name, 0) + 1

    # After processing, store the repo counts in a CSV
    repo_counts_df = pd.DataFrame(list(repo_counts.items()), columns=['Repository', 'Count'])
    repo_counts_df.to_csv('repository_counts.csv', index=False)

    # Create and return the Dataset object
    print("Total functions read till now is", total_functions)
    del total_functions
    return data

#Training data for NonFed learning:
train_dataset = create_dataset_train("train","all")
train_data = Dataset.from_dict(train_dataset)
train_data.save_to_disk(train_dataset_dir)
num_train_rows = len(train_data)
print("Train data of generated with #functions", num_train_rows)
del train_data, train_dataset

# Training data for Fed learning:
for this_client in range(0,num_clients):
    this_client_train_dataset = create_dataset_train("train",this_client)
    this_client_train_data = Dataset.from_dict(this_client_train_dataset)
    this_client_train_dataset_dir = client_datasets_dir+"/client_"+str(this_client)
    this_client_train_data.save_to_disk(this_client_train_dataset_dir)
    num_train_rows = len(this_client_train_data)
    print("Client ",this_client, ":Train data of generated with #functions", num_train_rows)
    del this_client_train_data, this_client_train_dataset


test_dataset = create_dataset("test")
test_data = Dataset.from_dict(test_dataset)
test_data.save_to_disk(test_dataset_dir)
print("Test data generated")