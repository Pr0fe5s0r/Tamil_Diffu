import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the HuggingFaceFW/clean-wikipedia dataset with the 'ta' (Tamil) subset
dataset = load_dataset("HuggingFaceFW/clean-wikipedia", "ta", split="train")

# Concatenate all text entries into a single string
data = "\n".join(dataset["text"])

# Split into train and val (90/10 split)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Load the Alibaba-NLP/gte-multilingual-base tokenizer
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")

# Encode with the multilingual tokenizer
train_ids = tokenizer.encode(train_data, add_special_tokens=False)
val_ids = tokenizer.encode(val_data, add_special_tokens=False)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
