import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

def get_tokenized_dataset(tokenizer, dataset_name, seq_len=256):
    """
    Tokenize different datasets based on the specified dataset name.
    
    Args:
    tokenizer: The tokenizer used to encode the texts.
    dataset_name: The name of the dataset to be processed ('wiki', 'cnn').
    seq_len: Maximum length of the tokenized sequences.

    Returns:
    A tokenized and formatted dataset for use in PyTorch.
    """
    def tokenize_function(examples):
        return tokenizer(examples[text_field], return_tensors='pt', max_length=seq_len, padding=True, truncation=True)
    
    if dataset_name == 'wiki':
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
        text_field = "text"
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=[text_field])
    elif dataset_name == 'cnn':
        dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
        text_field = "article"
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=[text_field])
    else:
        raise ValueError("Unsupported dataset name")

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset