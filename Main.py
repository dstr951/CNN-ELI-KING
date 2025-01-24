import Train
import Utils
import Consts
from Layers.BatchNorm import BatchNormalization
from Layers.CNN import Conv2D
from Layers.FC import FullyConnected
from Layers.Flatten import Flatten
from Layers.MaxPulling import MaxPooling2D
from Model import Model
import Visualizations
import numpy as np



def main():
    # Define the architecture
    model = Model([
        Conv2D(input_channels=3, output_channels=8, kernel_size=3, stride=1, padding=1),  # Convolution
        BatchNormalization(num_features=8),  # Batch Normalization
        MaxPooling2D(pool_size=2, stride=2),  # Max Pooling
        Flatten(),  # Flatten the output
        #FullyConnected(input_size=14 * 14 * 8, output_size=128),  # Fully Connected Layer
        FullyConnected(input_tuple=(Consts.BATCH_SIZE, 1800), output_size=(Consts.BATCH_SIZE,128)),  # Fully Connected Layer
        FullyConnected(input_tuple=(Consts.BATCH_SIZE,128), output_size=(Consts.BATCH_SIZE,10))  # Output Layer (e.g., 10 classes for classification)
    ])
    trained_model = Train.train(model)
    X_validation, Y_validation = Utils.read_labeled_file(Consts.VALIDATION_PATH)
    # reshape for 32 rows, 32 columns, 3 channels RGB
    X_validation = np.reshape(X_validation, (1000, 32, 32, 3))
    Y_pred = trained_model.forward(X_validation)
    Visualizations.generate_visualizations(Y_validation, Y_pred)





if __name__=="__main__":
    main()