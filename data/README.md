# Getting Data:

1. Install the Hugging Face CLI:
```bash
pip install -U "huggingface_hub[cli]"
```
2. Login to Hugging Face:
```bash
huggingface-cli login
```

## Wikitext:
- Link: https://huggingface.co/datasets/Salesforce/wikitext
- Rows: 
    - Training: 1.8 Million Rows
    - Validation: 3.76k Rows
    - Test: 4.36k Rows
```python
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
```

```python
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
```

## FineWeb-Edu:
- Link: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Rows:
    - Training: 1.43 Billion Rows
    - Subsets: ~10 Million Rows

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "default")
```

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20")
```