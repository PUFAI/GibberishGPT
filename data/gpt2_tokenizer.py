def get_gpt2_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Tokenizes input text using the tokenizer
def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

# Tokenizes the entire dataset using multiprocessing.
def tokenize_dataset(dataset, tokenizer, num_cores):
    return dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True, num_proc=num_cores)
###############################################################
# Sample tokenizer code
# from transformers import OpenAIGPTTokenizer

# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# text = "Hello, how are you?"
# input_ids = tokenizer.encode(text, return_tensors="pt")

# print("Token IDs:", input_ids)
# print("Decoded text:", tokenizer.decode(input_ids[0]))