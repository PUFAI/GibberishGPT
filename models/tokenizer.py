def get_tokenizer():
    from transformers import OpenAIGPTTokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    return tokenizer

# Sample tokenizer code
# from transformers import OpenAIGPTTokenizer

# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# text = "Hello, how are you?"
# input_ids = tokenizer.encode(text, return_tensors="pt")

# print("Token IDs:", input_ids)
# print("Decoded text:", tokenizer.decode(input_ids[0]))