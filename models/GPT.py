# GPT token-level model
from data import get_wikitext_data
from models import get_tokenizer

tokenizer = get_tokenizer()
tokenizer.pad_token = tokenizer.unk_token

# print("Vocab size:", tokenizer.vocab_size)
# print("Unknown token:", tokenizer.unk_token)
# print("Pad token:", tokenizer.pad_token)

dataset = get_wikitext_data()

print(dataset["train"][4])

sample_text = dataset["train"][4]["text"]
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)

print("Original Text:", sample_text)
print("Tokenized:", tokens)
print("Token IDs:", token_ids)