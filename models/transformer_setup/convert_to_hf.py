import os
import sys
import torch
from transformers import PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import shard_checkpoint
import json
from typing import Dict, Any, Optional, List

base_folder = os.path.abspath("../..")
sys.path.append(base_folder)
from tokenization import get_tiktoken_tokenizer
from transformer import TransformerModel
from params import ModelConfig

class FlashAttentionConfig(PretrainedConfig):
    model_type = "flash_attention_lm"
    
    def __init__(
        self,
        vocab_size=50257,
        n_embd=512,
        n_head=8,
        n_layer=8,
        block_size=512,
        dropout=0.1,
        use_flash_attn=True,
        gradient_checkpointing=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(**kwargs)


class FlashAttentionForCausalLM(PreTrainedModel):
    config_class = FlashAttentionConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerModel(
            vocab_size=config.vocab_size,
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            num_layers=config.n_layer,
            max_seq_len=config.block_size,
            dropout_prob=config.dropout,
            use_gradient_checkpoint=config.gradient_checkpointing,
            use_flash_attn=config.use_flash_attn
        )
        
    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        return {"logits": logits, "loss": loss}
        
    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        return self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            max_seq_len=self.config.block_size,
            **kwargs
        )


def convert_to_huggingface_format(checkpoint_path: str, output_dir: str):
    tokenizer = get_tiktoken_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint['config']
    
    config = FlashAttentionConfig(
        vocab_size=vocab_size,
        n_embd=config_dict['n_embd'],
        n_head=config_dict['n_head'],
        n_layer=config_dict['n_layer'],
        block_size=config_dict['block_size'],
        dropout=config_dict['dropout'],
        use_flash_attn=config_dict['use_flash_attn'],
        gradient_checkpointing=config_dict['gradient_checkpointing']
    )
    
    hf_model = FlashAttentionForCausalLM(config)
    state_dict = checkpoint['model_state_dict']
    hf_model.model.load_state_dict(state_dict)
    os.makedirs(output_dir, exist_ok=True)
    config.save_pretrained(output_dir)
    torch.save(hf_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    hf_tokenizer = convert_tiktoken_to_hf(tokenizer, output_dir)
    hf_tokenizer.save_pretrained(output_dir)
    create_model_card(output_dir, config_dict)

    print(f"Model converted and saved to {output_dir}")


def convert_tiktoken_to_hf(tiktoken_tokenizer, output_dir):
    merges = []
    for piece, rank in sorted(tiktoken_tokenizer.mergeable_ranks.items(), key=lambda x: x[1]):
        merged_token = b''.join(piece).decode('utf-8', errors='replace')
        merges.append(merged_token)
    
    vocab = {}
    for token, id in tiktoken_tokenizer.encoder.items():
        try:
            token_str = token.decode('utf-8', errors='replace')
        except:
            token_str = str(token)
        vocab[token_str] = id
    
    tokenizer = PreTrainedTokenizerFast(
        vocab=vocab,
        merges=merges,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>"
    )
    
    return tokenizer


def create_model_card(output_dir, config_dict):
    """Create a model card markdown file"""
    model_card = f"""---
        language: en
        license: mit
        tags:
        - pytorch
        - causal-lm
        - language-model
        - flash-attention
        ---

        # Flash Attention Language Model

        This is a language model trained with Flash Attention. The model is based on a decoder-only transformer architecture.

        ## Model Details

        - **Model Type:** Causal Language Model
        - **Training Data:** Wikitext
        - **Embedding Size:** {config_dict['n_embd']}
        - **Hidden Layers:** {config_dict['n_layer']}
        - **Attention Heads:** {config_dict['n_head']}
        - **Context Length:** {config_dict['block_size']}
        - **Flash Attention:** {'Enabled' if config_dict['use_flash_attn'] else 'Disabled'}

        ## Usage

        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("PurelyUnfunctionalAI/GibberishGPT")
        model = AutoModelForCausalLM.from_pretrained("PurelyUnfunctionalAI/GibberishGPT")

        input_text = "Your prompt here"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=50)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)
        ```

        ## License

        This model is available under the MIT License.
        """
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert model to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for converted model")
    
    args = parser.parse_args()
    
    convert_to_huggingface_format(args.checkpoint, args.output_dir)
