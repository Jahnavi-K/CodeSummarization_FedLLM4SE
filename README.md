# CodeSummarization_FedLLM4SE
Code Summarization without Direct Access to Code - Towards Exploring Federated LLMs for Software Engineering. The corresponding paper has been accepted in EASE'24, for further details on the code, methodology, and results, please refer <doi>

# **CodeSummarization_FedLLM4SE**

## **Code Summarization without Direct Access to Code - Towards Exploring Federated LLMs for Software Engineering**

"Code Summarization without Direct Access to Code - Towards Exploring Federated LLMs for Software Engineering" is a exploration study conducted in 2024. The study PEFT LoRA fine-tune an open source model Llama2 using python dataset https://github.com/EdinburghNLP/code-docstring-corpus. The federated approach is followed in training the model using 3 clients. 

## **MODEL**
Trained model is available at [hugging face](https://huggingface.co/JahnaviKumar/7BCodeLLama_PyCdSmry_Hetro_Central_LoRA/tree/main)

## **ARTIFACTS**
Fed Adapter artifacts are available at [OSF](https://osf.io/wnq8s/?view_only=c7e2a810bc1e4b5dada5e3336c92da01)

The terms "centrally-trained model" and "non-federated model" are often used interchangeably and essentially represent the same concept. This work was carried out from Q3 of the year 2023.
## **CODE**
Code folder contains py files for:

- First file is for splitting the dataset to clients for federated simulation. Each client gets few repositories.
- Next 3 seraches the space for target attention modules, rank of LoRA, optimal Training Data size.
- Then, Central(Non-federated) model is trained.
- Finally, Federated rounds are initiated with round 0 in first file, then rest rounds are handled by last file.

## **ModelAdapters folder contains LoRA adapters:**
Because the model size is huge, we are providing adapters which can be merged with base model for obtaining the fine-tuned model.

- NonFed folder has Adapters created during hyperparameter search of target module (W), rank of LoRA(r), optimal Training Data(TD) size. Finally, the Central model created using chosen hyperparameters.
- round0, round1, ..., round20 represent the Federated Learning round. Each folder has sub-folder for server and each client: client0,client1,client2. Respective adapters are inside Ada subfolders.

## **Results folder contains csv files contains metrics for experiments related to Code Summarization:**

- First 3 files are the results during hyperparameter search of target module (W), rank of LoRA(r), optimal Training Data(TD) size.
- The last file is repesenting Centrally(Non-Fed) trained model along with 20 Federated rounds' results.

## **Replication steps:**

- Download the dataset from https://github.com/EdinburghNLP/code-docstring-corpus/tree/master/repo_split.parallel-corpus
- Execute first file in Code folder to split data into repositories. Please feel free to experiment with configurtaion files LLMbase_config.py and FedCS_LLMinput.py.
- Select any 10k random training data to find hyperparameters using 2nd to 4th code file.
- Update the optimal hyperparameters in the configurtaion files LLMbase_config.py and FedCS_LLMinput.py and train central model(non-fed model) using 5th code file.
- Execute 6th file to setup for fed training. Lastly run 7th file to have the fed rounds generated. The models will be evaluated and results will be created as csv files.
