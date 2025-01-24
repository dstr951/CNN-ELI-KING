from typing import Tuple

import numpy as np
import Consts
def read_test_file(path:str) -> Tuple[np.array,np.array]:
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


