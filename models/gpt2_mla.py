import torch
import multiprocessing

from data import get_wikitext_data, get_tiktoken_tokenizer, tiktoken_tokenize_dataset, save_data, load_data

batch_size = 64  # Kept the same; could be adjusted based on hardware
block_size = 1024  # GPT-2 uses a context length of 1024 tokens
max_iters = 50000  # More iterations needed for larger models
eval_interval = 1000  # Increase since more iterations are done
learning_rate = 5e-5  # GPT-2 uses a lower learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500  # More frequent evaluation for stability

n_embd = 768  # GPT-2 uses 768 for the small version, 1024 for medium, 1280 for large, 1600 for XL
n_head = 12  # GPT-2 uses 12 attention heads
n_layer = 12  # GPT-2 has 12 transformer blocks in the small version
dropout = 0.1  # GPT-2 uses 0.1 dropout for better generalization

print(device)

DATA_PATH = "/workspace/GPT/data/tiktoken_tokenized_wikitext"
num_cores = multiprocessing.cpu_count()

tokenizer = get_tiktoken_tokenizer()
dataset = get_wikitext_data()

# tokenized_dataset = tiktoken_tokenize_dataset(dataset, tokenizer, num_cores)
# print(tokenized_dataset)

tokenized_dataset = load_data(DATA_PATH)

def convert_to_tensor(example):
    return {"tokens": torch.tensor(example["input_ids"], dtype=torch.long)}

tokenized_dataset = tokenized_dataset.map(convert_to_tensor, batched=False)