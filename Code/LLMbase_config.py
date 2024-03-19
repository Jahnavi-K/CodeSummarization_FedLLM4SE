# The model that you want to train from the Hugging Face hub
model_id = "NousResearch/Llama-2-7b-hf"
# Load the entire model on the GPU 0
device_map = {"": 0}
#device_map={"":torch.cuda.current_device()}
# bitsandbytes parameters
# Activate 4-bit precision base model loading
use_4bit = False
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float32"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "fp4"
# Activate nested quantization for 4-bit base models (double quantization)
use_double_nested_quant = False

#QLoRA parameters
# LoRA attention dimension
lora_r = 8  #16#64
LORA_TARGET_MODULES = ["q_proj","k_proj"]
# Alpha parameter for LoRA scaling
lora_alpha = 16 #lora_r*2#
# Dropout probability for LoRA layers
lora_dropout = 0.1


#Training argument parameters
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 6 #2#6
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1 #1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 #1e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "adamw_torch" #paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant" cosine
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False #cant be true coz..ValueError: the `--group_by_length` option is only available for `Dataset`, not `IterableDataset
# Save checkpoint every X updates steps
save_steps = -1# 1000 #0 updated to 1000
# Log every X updates steps
logging_steps = 25
# Disable tqdm
disable_tqdm= True
max_split_size_mb=2000  # Set this value based on your GPU's memory capacity

#SFTT Trainer parameters
# Maximum sequence length to use
max_seq_length = 2048 #None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True #False