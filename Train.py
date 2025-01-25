from typing import List, Tuple

import numpy as np

import Model
import Consts
import Utils


def min_max_norm(X):
    min_values = np.min(X, axis = 0)
    max_values = np.max(X, axis = 0)
    EPSILON = 1e-8
    return (X - min_values) / (max_values - min_values + EPSILON)


def preprocess_data() -> List[Tuple[np.array, np.array]]:
    X, Y = Utils.read_labeled_file(Consts.TRAIN_PATH_CFAR)
    X_validate, Y_valdiate = Utils.read_labeled_file(Consts.VALIDATION_PATH)
    X, Y = shuffle_data(X, Y)
    X = min_max_norm(X)
    X_train, Y_train = X[:24000], Y[:24000]
    X_validate, Y_valdiate = X[24000:25000], Y[24000:25000]
    # reshape for 32 rows, 32 columns, 3 channels RGB
    X_train = np.reshape(X_train, (24000, 32, 32, 3))
    X_validate = np.reshape(X_validate, (1000, 32, 32, 3))
    # TODO - add minmax normalization to data

    return create_mini_batches(X_train, Y_train, Consts.BATCH_SIZE), X_validate, Y_valdiate


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

    # Create batches
    mini_batches = [
        (X[i:i + batch_size], Y[i:i + batch_size])
        for i in range(0, n_samples, batch_size)
    ]

    return mini_batches

def train(model: Model):
    batches, X_validate, Y_valdiate = preprocess_data()
    for epoch in range(Consts.NUM_EPOCHS):
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print(f"Epoch {epoch + 1}/{Consts.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(batches):
            X_batch = batch[0]
            Y_batch = batch[1]
            Y_batch = Y_batch.astype(int)
            Y_BATCH_ONE_HOT = np.eye(10)[Y_batch - 1]

            predictions = model.forward(X_batch)

            # Compute loss
            loss = loss_fn(predictions, Y_BATCH_ONE_HOT)
            train_loss += loss

            # Compute gradient of loss with respect to predictions
            loss_grad = compute_loss_gradient(predictions, Y_BATCH_ONE_HOT)
            model.backward(loss_grad)

            # Update model parameters
            model.update_params(Consts.LEARNING_RATE)

            # Calculate batch accuracy
            batch_predictions = np.argmax(predictions, axis=1)
            correct_predictions += np.sum(batch_predictions == Y_batch)
            total_samples += Y_batch.shape[0]

            # Print batch progress
            print(f"\tBatch {batch_idx + 1}/{len(batches)} - Loss: {loss:.4f}")

        # Print epoch accuracy and loss


        predictions = model.forward(X_validate)
        validation_predictions = np.argmax(predictions, axis=1)
        correct_predictions += np.sum(validation_predictions == Y_valdiate)
        total_samples = Y_valdiate.shape[0]
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss / len(batches):.4f}, Accuracy: {epoch_accuracy:.4f}")

    return model, X_validate, Y_valdiate

def loss_fn(predictions, Y_batch):
    return categorical_cross_entropy(predictions, Y_batch)

def categorical_cross_entropy(predictions, targets):
    """
    Computes the Categorical Cross-Entropy Loss.

    Args:
        predictions (np.ndarray): Softmax output of shape (N, C), where N is the batch size,
                                   and C is the number of classes. Each row should sum to 1.
        targets (np.ndarray): One-hot encoded ground truth of shape (N, C).

    Returns:
        float: Average cross-entropy loss over the batch.
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    # Compute the loss
    loss = -np.sum(targets * np.log(predictions)) / targets.shape[0]
    return loss

def compute_loss_gradient(predictions, targets):
    """
    Computes the gradient of the Categorical Cross-Entropy Loss with respect to predictions.

    Args:
        predictions (np.ndarray): Softmax probabilities of shape (N, C), where N is the batch size
                                   and C is the number of classes.
        targets (np.ndarray): One-hot encoded ground truth of shape (N, C).

    Returns:
        np.ndarray: Gradient of the loss with respect to predictions, of shape (N, C).
    """
    # Ensure shapes match
    assert predictions.shape == targets.shape, "Shapes of predictions and targets must match."

    # Gradient is simply (predictions - targets)
    return predictions - targets
