import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_wake_word_dataset(
    target_word: str,
    n_positives: int = 3000,
    n_negatives: int = 7000,
    input_dim: int = 1536,
    batch_size: int = 64
) -> DataLoader:
    """
    Synthesizes positive target embeddings and negative ambient noise
    embeddings, returning a PyTorch DataLoader.
    """
    # Deterministic acoustic pattern tied to the phrase string
    seed = abs(hash(target_word)) % (2**32)
    rng = np.random.default_rng(seed)
    
    # Target word acoustic signature
    word_signature = rng.normal(loc=0.6, scale=0.15, size=(input_dim,)).astype(np.float32)

    # Positive samples: target signature + phonetic and background noise variations
    pos_samples = word_signature + rng.normal(loc=0.0, scale=0.25, size=(n_positives, input_dim)).astype(np.float32)
    pos_labels = np.ones((n_positives, 1), dtype=np.float32)

    # Negative samples: background chatter, noise, unaligned phonemes
    neg_samples = rng.normal(loc=-0.1, scale=0.4, size=(n_negatives, input_dim)).astype(np.float32)
    neg_labels = np.zeros((n_negatives, 1), dtype=np.float32)

    # Merge and randomize indices
    x_data = np.vstack([pos_samples, neg_samples])
    y_data = np.vstack([pos_labels, neg_labels])
    
    indices = rng.permutation(len(x_data))
    x_tensor = torch.tensor(x_data[indices])
    y_tensor = torch.tensor(y_data[indices])

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)