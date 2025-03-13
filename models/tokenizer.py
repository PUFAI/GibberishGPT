def get_tokenizer():
    from transformers import OpenAIGPTTokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    return tokenizer

def get_gpt2_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_tiktoken_tokenizer(model_name="gpt-2"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding

# Tokenizes input text using the tokenizer
def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

# Tokenizes the entire dataset using multiprocessing.
def tokenize_dataset(dataset, tokenizer, num_cores):
    return dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True, num_proc=num_cores)

# Tokenizes input text using the tiktoken encoder
def tiktoken_tokenize_function(tokenizer, examples):
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

# Tokenizes the entire dataset using multiprocessing using tiktoken tokenizer
def tiktoken_tokenize_dataset(dataset, tokenizer, num_cores):
    return dataset.map(lambda x: tiktoken_tokenize_function(tokenizer, x), batched=True, num_proc=num_cores)
###############################################################
# Sample tokenizer code
# from transformers import OpenAIGPTTokenizer

# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# text = "Hello, how are you?"
# input_ids = tokenizer.encode(text, return_tensors="pt")

# print("Token IDs:", input_ids)
# print("Decoded text:", tokenizer.decode(input_ids[0]))