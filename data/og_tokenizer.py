def get_tokenizer():
    from transformers import OpenAIGPTTokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    return tokenizer