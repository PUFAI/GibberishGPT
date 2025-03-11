def get_fineweb_data():
    from datasets import load_dataset
    fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default")
    return fineweb_dataset