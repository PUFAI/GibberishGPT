# GPT token-level model
from data import get_wikitext_data
from datasets import load_from_disk
from models import get_gpt2_tokenizer

import multiprocessing

num_cores = multiprocessing.cpu_count()

tokenizer = get_gpt2_tokenizer()
tokenizer.pad_token = tokenizer.eos_token

dataset = get_wikitext_data()

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=num_cores)
print(tokenized_datasets["train"][4])

tokenized_datasets.save_to_disk("/workspace/GPT/data/tokenized_wikitext")
saved_datasets = load_from_disk("/workspace/GPT/data/tokenized_wikitext")
print(saved_datasets["train"][4])

############################################################
# Example tokenization

# from models import get_tokenizer

# tokenizer = get_tokenizer()
# tokenizer.pad_token = tokenizer.unk_token

# print("Vocab size:", tokenizer.vocab_size)
# print("Unknown token:", tokenizer.unk_token)
# print("Pad token:", tokenizer.pad_token)

# dataset = get_wikitext_data()

# print(dataset["train"][4])

# sample_text = dataset["train"][4]["text"]
# tokens = tokenizer.tokenize(sample_text)
# token_ids = tokenizer.encode(sample_text)

# print("Original Text:", sample_text)
# print("Tokenized:", tokens)
# print("Token IDs:", token_ids)