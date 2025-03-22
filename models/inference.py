import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)
from tokenization import get_tiktoken_tokenizer
tokenizer = get_tiktoken_tokenizer()
vocab_size = tokenizer.n_vocab

class Head(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        keys = self.key_proj(input_tensor)
        queries = self.query_proj(input_tensor)
        values = self.value_proj(input_tensor)
        attn_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        attn_scores = attn_scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ values
        return output

class FlashAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN

    def forward(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        keys = self.key_proj(input_tensor)
        queries = self.query_proj(input_tensor)
        values = self.value_proj(input_tensor)
        
        if self.use_flash and seq_len <= 1024:
            q = queries.unsqueeze(2)
            k = keys.unsqueeze(2)
            v = values.unsqueeze(2)
            output = flash_attn_func(q, k, v, causal=True)
            output = output.squeeze(2)
        else:
            attn_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
            attn_scores = attn_scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = attn_weights @ values
        return output

class MultiHead(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn=False):
        super().__init__()
        head_class = FlashAttentionHead if (HAS_FLASH_ATTN and use_flash_attn) else Head
        self.heads = nn.ModuleList([
            head_class(embed_dim, head_dim, max_seq_len, dropout_prob)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_tensor):
        head_outputs = [head(input_tensor) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        projected = self.projection(concatenated)
        return self.dropout(projected)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w2 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w3 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_tensor):
        swish = self.w1(input_tensor) * torch.sigmoid(self.w1(input_tensor))
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn=False):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.self_attention = MultiHead(num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn)
        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input_tensor):
        x = input_tensor + self.self_attention(self.layer_norm1(input_tensor))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class TransformerModel(nn.Module):
    """Transformer-based language model for inference"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob,
                 use_gradient_checkpoint=False, use_flash_attn=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        token_emb = self.token_embedding(idx)
        positions = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * seq_len, -1)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    print("Checkpoint loaded. Model configuration:")
    for key, val in config_dict.items():
        print(f"  {key}: {val}")

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=config_dict['n_embd'],
        num_heads=config_dict['n_head'],
        num_layers=config_dict['n_layer'],
        max_seq_len=config_dict['block_size'],
        dropout_prob=config_dict['dropout'],
        use_gradient_checkpoint=config_dict['gradient_checkpointing'],
        use_flash_attn=config_dict['use_flash_attn']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("checkpoints", "best_model.pt")
    model = load_model(checkpoint_path, device)

    print("\nEnter your prompt below. Type 'exit' to quit.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower().strip() == "exit":
            break
        prompt_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated_tensor = model.generate(input_tensor, max_new_tokens=200, temperature=1.0)
        generated_text = tokenizer.decode(generated_tensor[0].tolist())
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main()
