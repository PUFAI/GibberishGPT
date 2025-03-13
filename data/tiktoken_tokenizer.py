def get_tiktoken_tokenizer(model_name="gpt-2"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding

# Tokenizes input text using the tiktoken encoder
def tiktoken_tokenize_function(tokenizer, examples):
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

# Tokenizes the entire dataset using multiprocessing using tiktoken tokenizer
def tiktoken_tokenize_dataset(dataset, tokenizer, num_cores):
    return dataset.map(lambda x: tiktoken_tokenize_function(tokenizer, x), batched=True, num_proc=num_cores)