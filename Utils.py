from typing import Tuple

import numpy as np
import Consts
def read_labeled_file(path:str) -> Tuple[np.array, np.array]:
    # Load the CSV data into a NumPy array
    # No headers, so skiprows is not needed
    data = np.loadtxt(path, delimiter=',')

    # Split the data into Y (labels) and X (features)
    Y = data[:, 0]  # First column (Y)
    X = data[:, 1:]  # All remaining columns (X)


    # Print shapes to confirm
    print("Y shape:", Y.shape)
    print("X shape:", X.shape)
    return X,Y


def to_img(V):
    n_samples = V.shape[0]

    # Reshape into (n_samples, 32, 32, 3)
    V = np.reshape(V, (n_samples, 3, 32, 32))  # Temporary shape for channel-first
    V = np.transpose(V, (0, 2, 3, 1))  # Convert to channel-last (n_samples, 32, 32, 3)
    return V

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