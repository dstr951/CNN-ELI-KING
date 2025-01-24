from typing import List, Tuple

import numpy as np

import Model
import Consts
import Utils

def preprocess_data() -> List[Tuple[np.array, np.array]]:
    X, Y = Utils.read_test_file(Consts.TRAIN_PATH)
    # reshape for 32 rows, 32 columns, 3 channels RGB
    X = np.reshape(X, (8000, 32, 32, 3))
    # TODO - add minmax normalization to data

    return create_mini_batches(X, Y, Consts.BATCH_SIZE)


# Function to create mini-batches with a random seed
def create_mini_batches(X, Y, batch_size, seed=Consts.SEED):
    """
    Splits X and Y into mini-batches with optional random seed.

    Args:
        X (ndarray): Features, shape (n_samples, n_features).
        Y (ndarray): Labels, shape (n_samples,).
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for reproducibility (optional).

    Returns:
        List of tuples (X_batch, Y_batch)
    """
    n_samples = X.shape[0]

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Shuffle the data
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Create batches
    mini_batches = [
        (X[i:i + batch_size], Y[i:i + batch_size])
        for i in range(0, n_samples, batch_size)
    ]

    return mini_batches

def train(model: Model):
    batches = preprocess_data()
    for epoch in range(Consts.NUM_EPOCHS):
        for batch in batches:
            X_batch = batch[0]
            Y_batch = batch[1]
            model.forward(X_batch)
        print(f"finished epoch: {epoch}")
    return model

