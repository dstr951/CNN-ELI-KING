import copy
from typing import List, Tuple
import Model
import Consts
import Utils

import numpy as np

def rotate_image_numpy(image, angle):
    """
    Rotates an image by a specified angle using only NumPy.

    Args:
        image (np.ndarray): The input image of shape (height, width, channels).
        angle (float): The angle (in degrees) to rotate the image.

    Returns:
        np.ndarray: The rotated image of the same shape as the input.
    """
    height, width, channels = image.shape
    center_x, center_y = width // 2, height // 2  # Calculate the center of the image

    # Create the rotation matrix
    rad_angle = np.deg2rad(angle)
    cos_theta = np.cos(rad_angle)
    sin_theta = np.sin(rad_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Create an output image
    rotated_image = np.zeros_like(image)

    # Iterate through each pixel in the output image
    for y in range(height):
        for x in range(width):
            # Map the pixel in the rotated image back to the original image
            original_coords = np.array([x - center_x, y - center_y])
            new_coords = np.dot(rotation_matrix, original_coords)
            new_x, new_y = new_coords + np.array([center_x, center_y])

            # Check if the coordinates are within bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                # Assign pixel value from the original image to the rotated image
                rotated_image[y, x] = image[int(new_y), int(new_x)]

    return rotated_image


def augment_with_rotation_numpy(X, max_angle=5):
    """
    Augments the dataset by rotating each image by a random degree between -max_angle and max_angle.

    Args:
        X (np.ndarray): Original dataset of images, shape (n_samples, height, width, channels).
        max_angle (int): Maximum absolute angle to rotate the images.

    Returns:
        np.ndarray: Augmented dataset with rotated images, same shape as input.
    """
    augmented_X = np.zeros_like(X)

    for i in range(X.shape[0]):
        angle = np.random.uniform(-max_angle, max_angle)  # Random angle between -max_angle and max_angle
        augmented_X[i] = rotate_image_numpy(X[i], angle)

    return augmented_X




def show_image_and_channels(matrix):
    import matplotlib.pyplot as plt
    """
    Displays an image represented by a 32x32x3 NumPy matrix, along with its individual RGB channels.

    Args:
        matrix (np.ndarray): A 32x32x3 NumPy array representing an image.
    """
    assert matrix.shape == (32, 32, 3), "Input matrix must have shape (32, 32, 3)."

    # Extract individual channels
    red_channel = matrix[:, :, 0]   # Red channel
    green_channel = matrix[:, :, 1] # Green channel
    blue_channel = matrix[:, :, 2]  # Blue channel

    # Plot the channels and the full image
    plt.figure(figsize=(10, 10))

    # Red channel
    plt.subplot(2, 2, 1)
    plt.title("Red Channel")
    plt.imshow(red_channel, cmap="Reds")
    plt.axis("off")

    # Green channel
    plt.subplot(2, 2, 2)
    plt.title("Green Channel")
    plt.imshow(green_channel, cmap="Greens")
    plt.axis("off")

    # Blue channel
    plt.subplot(2, 2, 3)
    plt.title("Blue Channel")
    plt.imshow(blue_channel, cmap="Blues")
    plt.axis("off")

    # Full image
    plt.subplot(2, 2, 4)
    plt.title("Full Image")
    plt.imshow(matrix)  # Matplotlib automatically handles RGB
    plt.axis("off")

    # Show the plots
    plt.tight_layout()
    plt.show()


def preprocess_data() -> List[Tuple[np.array, np.array]]:
    # Load data
    X, Y = Utils.read_labeled_file(Consts.TRAIN_PATH)
    X_validate, Y_validate = Utils.read_labeled_file(Consts.VALIDATION_PATH)

    # Number of images
    n_samples = X.shape[0]

    # Reshape into (n_samples, 32, 32, 3)
    X = np.reshape(X, (n_samples, 3, 32, 32))  # Temporary shape for channel-first
    X = np.transpose(X, (0, 2, 3, 1))          # Convert to channel-last (n_samples, 32, 32, 3)
    # show_image_and_channels(X[0])              # Visualize the first image

    # Reshape validation set similarly
    n_validate_samples = X_validate.shape[0]
    X_validate = np.reshape(X_validate, (n_validate_samples, 3, 32, 32))
    X_validate = np.transpose(X_validate, (0, 2, 3, 1))

    # Combine data for augmentation
    X_combined = X
    Y_combined = Y
    for i in range(2):
        # Apply data augmentation with rotation using NumPy
        X_augmented = augment_with_rotation_numpy(X)

        # Combine the original and augmented datasets
        X_combined = np.concatenate([X_combined, X_augmented], axis=0)
        Y_combined = np.concatenate([Y_combined, Y], axis=0)  # Duplicate labels for augmented images

    # Shuffle the combined dataset
    X_combined, Y_combined = shuffle_data(X_combined, Y_combined)

    return create_mini_batches(X_combined, Y_combined, Consts.BATCH_SIZE), X_validate, Y_validate



def shuffle_data(X, Y):
    n_samples = X.shape[0]

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y


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

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Create batches
    mini_batches = [
        (X[i:i + batch_size], Y[i:i + batch_size])
        for i in range(0, n_samples, batch_size)
    ]

    return mini_batches


def train(model: Model):
    batches,  X_validate, Y_validate = preprocess_data()
    max_epoch_accuracy = -1
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
            batch_predictions = np.argmax(predictions, axis=1) + 1
            correct_predictions += np.sum(batch_predictions == Y_batch)
            total_samples += Y_batch.shape[0]

            # Print batch progress
            if batch_idx % 20 == 19:
                print(f"\tBatch {batch_idx + 1}/{len(batches)} - Loss: {loss:.4f}")

        # Print epoch accuracy and loss

        train_accuracy = correct_predictions / total_samples
        validation_predictions = model.inference(X_validate)
        correct_predictions = np.sum(validation_predictions == Y_validate)
        total_samples = Y_validate.shape[0]
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss / len(batches):.4f}, validation Accuracy: {epoch_accuracy:.4f} train Accuracy: {train_accuracy:.4f}")
        if max_epoch_accuracy < epoch_accuracy:
            max_epoch_accuracy = epoch_accuracy
            max_model = copy.deepcopy(model)
    print(f"returning model with accuracy: {max_epoch_accuracy}")
    return max_model, X_validate, Y_validate


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
