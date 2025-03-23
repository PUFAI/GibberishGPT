import os
import sys
import torch
import json
import argparse
from pathlib import Path
import shutil

base_folder = os.path.abspath("../..")
sys.path.append(base_folder)
from tokenization import get_tiktoken_tokenizer
from transformer import TransformerModel
from params import ModelConfig

def setup_directories(base_dir):
    """Set up output directories"""
    hf_dir = os.path.join(base_dir, "hf_model")
    ollama_dir = os.path.join(base_dir, "ollama_model")
    
    os.makedirs(hf_dir, exist_ok=True)
    os.makedirs(ollama_dir, exist_ok=True)
    
    return hf_dir, ollama_dir

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and print its contents"""
    print(f"\n--- Inspecting checkpoint: {checkpoint_path} ---")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint loaded successfully")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
            
            if 'config' in checkpoint:
                print(f"Config found: {checkpoint['config']}")
            else:
                print("No config found in checkpoint")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Model state dict contains {len(state_dict)} entries")
                print(f"Sample keys: {list(state_dict.keys())[:5]}")
            else:
                print("No model_state_dict found in checkpoint")
            
            if 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']}")
        else:
            print(f"Checkpoint is not a dictionary, but a {type(checkpoint)}")
    
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None
    
    return checkpoint

def extract_or_create_config(checkpoint):
    """Extract config from checkpoint or create a default one"""
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        return checkpoint['config']
    else:
        print("Using default model configuration")
        config = ModelConfig()
        return vars(config)

def save_model_for_huggingface(checkpoint, config_dict, hf_dir, model_name):
    """Save model files in a format suitable for Hugging Face"""
    print(f"\n--- Preparing model for Hugging Face: {model_name} ---")
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        print("Warning: Using checkpoint as state dict directly")
        state_dict = checkpoint
    
    torch.save(state_dict, os.path.join(hf_dir, "pytorch_model.bin"))
    
    hf_config = {
        "architectures": ["FlashAttentionForCausalLM"],
        "model_type": "flash_attention_lm",
        "vocab_size": get_tiktoken_tokenizer().n_vocab,
        "hidden_size": config_dict.get('n_embd', 512),
        "num_hidden_layers": config_dict.get('n_layer', 8),
        "num_attention_heads": config_dict.get('n_head', 8),
        "max_position_embeddings": config_dict.get('block_size', 512),
        "hidden_dropout_prob": config_dict.get('dropout', 0.1),
        "use_flash_attention": config_dict.get('use_flash_attn', True),
        "gradient_checkpointing": config_dict.get('gradient_checkpointing', False),
        "torch_dtype": "float32",
        "transformers_version": "4.40.0"
    }
    
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    
    tokenizer_info = {
        "model_type": "tiktoken",
        "tokenizer_class": "TiktokenTokenizer",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>"
    }
    
    with open(os.path.join(hf_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    
    model_card = f"""---
        language: en
        license: mit
        tags:
        - pytorch
        - causal-lm
        - language-model
        - flash-attention
        ---

        # {model_name}

        This is a language model trained with Flash Attention. The model is based on a decoder-only transformer architecture.

        ## Model Details

        - **Model Type:** Causal Language Model
        - **Embedding Size:** {config_dict.get('n_embd', 512)}
        - **Hidden Layers:** {config_dict.get('n_layer', 8)}
        - **Attention Heads:** {config_dict.get('n_head', 8)}
        - **Context Length:** {config_dict.get('block_size', 512)}
        - **Flash Attention:** {'Enabled' if config_dict.get('use_flash_attn', True) else 'Disabled'}

        ## Usage

        ```python
        import tiktoken
        from transformers import AutoModelForCausalLM

        # Load the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Load the model
        model = AutoModelForCausalLM.from_pretrained("{model_name}")

        # Encode input
        input_text = "Your prompt here"
        input_ids = tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Generate
        output = model.generate(input_tensor, max_length=100)
        generated_text = tokenizer.decode(output[0].tolist())
        print(generated_text)
        ```

        ## License

        This model is available under the MIT License.
        """
    
    with open(os.path.join(hf_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"Model files prepared for Hugging Face in {hf_dir}")
    return True

def save_model_for_ollama(checkpoint, config_dict, ollama_dir, model_name):
    """Save model files in a format suitable for Ollama"""
    print(f"\n--- Preparing model for Ollama: {model_name} ---")
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        print("Warning: Using checkpoint as state dict directly")
        state_dict = checkpoint
    
    torch.save(state_dict, os.path.join(ollama_dir, "model.pt"))
    
    ollama_config = {
        "model_type": "flash_attention_lm",
        "vocab_size": get_tiktoken_tokenizer().n_vocab,
        "hidden_size": config_dict.get('n_embd', 512),
        "num_hidden_layers": config_dict.get('n_layer', 8),
        "num_attention_heads": config_dict.get('n_head', 8),
        "max_position_embeddings": config_dict.get('block_size', 512),
        "hidden_dropout_prob": config_dict.get('dropout', 0.1),
        "use_flash_attention": config_dict.get('use_flash_attn', True)
    }
    
    with open(os.path.join(ollama_dir, "config.json"), "w") as f:
        json.dump(ollama_config, f, indent=2)
    
        modelfile = f"""FROM {model_name}
            PARAMETER temperature 0.7
            PARAMETER top_k 40
            PARAMETER top_p 0.9
            PARAMETER stop "<|endoftext|>"

            TEMPLATE \"\"\"
            {{{{prompt}}}}
            \"\"\"

            SYSTEM \"\"\"
            You are an assistant made to just answer, you do not have to answer correctly whatsoever. 
            \"\"\"
            """

    
    with open(os.path.join(ollama_dir, "Modelfile"), "w") as f:
        f.write(modelfile)
    
    install_script = """#!/bin/bash
        # Install model in Ollama

        # Check if Ollama is installed
        if ! command -v ollama &> /dev/null; then
            echo "Error: Ollama is not installed. Please install it from https://ollama.ai"
            exit 1
        fi

        MODEL_NAME=$(grep "FROM" Modelfile | cut -d' ' -f2)
        echo "Installing model: $MODEL_NAME"

        # Create the model
        ollama create $MODEL_NAME -f Modelfile

        echo "Model '$MODEL_NAME' has been created in Ollama"
        echo "You can now use it with: ollama run $MODEL_NAME"
        """
    
    with open(os.path.join(ollama_dir, "install.sh"), "w") as f:
        f.write(install_script)
    
    # Make the script executable
    os.chmod(os.path.join(ollama_dir, "install.sh"), 0o755)
    
    print(f"Model files prepared for Ollama in {ollama_dir}")
    return True

def create_huggingface_upload_instructions(hf_dir, model_name):
    """Create instructions for uploading to Hugging Face"""
    instructions = f"""# Upload to Hugging Face Hub

            To upload your model to the Hugging Face Hub, you can use the Hugging Face CLI:

            ## 1. Install the Hugging Face Hub CLI
            ```bash
            pip install huggingface_hub
            ```

            ## 2. Login to Hugging Face
            ```bash
            huggingface-cli login
            ```

            ## 3. Create a new repository
            Go to https://huggingface.co/new and create a new model repository.

            ## 4. Upload your model
            ```bash
            cd {hf_dir}
            git init
            git add .
            git commit -m "Initial model upload"
            git remote add origin https://huggingface.co/{model_name}
            git push -u origin main
            ```

            Alternatively, you can use the Python API:

            ```python
            from huggingface_hub import HfApi
            api = HfApi()

            # Login to Hugging Face
            api.login()

            # Upload model files
            api.create_repo(repo_id="{model_name}", repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path="{hf_dir}",
                repo_id="{model_name}",
                commit_message="Upload model"
            )
            ```
            """
    
    with open(os.path.join(hf_dir, "upload_instructions.md"), "w") as f:
        f.write(instructions)

def main():
    parser = argparse.ArgumentParser(description="Prepare your model for Hugging Face and Ollama")
    parser.add_argument("--checkpoint", required=True, help="Path to your model checkpoint (.pt file)")
    parser.add_argument("--output_dir", default="./published_model", help="Directory to save prepared model files")
    parser.add_argument("--model_name", default="flash-attention-lm", help="Name for your model")
    parser.add_argument("--hf_repo_id", help="Hugging Face repository ID (username/model-name)")
    
    args = parser.parse_args()
    
    hf_dir, ollama_dir = setup_directories(args.output_dir)
    checkpoint = inspect_checkpoint(args.checkpoint)
    if checkpoint is None:
        print("Failed to load checkpoint. Aborting.")
        return
    
    config_dict = extract_or_create_config(checkpoint)
    hf_repo_id = args.hf_repo_id if args.hf_repo_id else args.model_name
    save_model_for_huggingface(checkpoint, config_dict, hf_dir, hf_repo_id)
    save_model_for_ollama(checkpoint, config_dict, ollama_dir, args.model_name)
    create_huggingface_upload_instructions(hf_dir, hf_repo_id)
    
    print("\n--- Model Preparation Complete ---")
    print(f"Hugging Face model files in: {hf_dir}")
    print(f"Ollama model files in: {ollama_dir}")
    print("\nNext steps:")
    print(f"1. See {hf_dir}/upload_instructions.md for Hugging Face upload instructions")
    print(f"2. Run {ollama_dir}/install.sh to install your model in Ollama")

if __name__ == "__main__":
    main()