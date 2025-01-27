import Train
import Utils
import Consts
from Layers.BatchNorm import BatchNormalization
from Layers.CNN import Conv2D
from Layers.Dropout import Dropout
from Layers.FC import FullyConnected
from Layers.Flatten import Flatten
from Layers.MaxPulling import MaxPooling2D
from Layers.Relu import ReLU
from Layers.Softmax import Softmax
from Model import Model
import Visualizations
import numpy as np
# Set a seed for reproducibility
np.random.seed(42)

def cnn_dims(h, w, k, p, s, c_in, c_out):
    """
    Compute the output height, width, and number of channels for a convolutional layer.

    Args:
    - h (int): input height
    - w (int): input width
    - k (int): kernel size (assuming square kernel)
    - p (int): padding
    - s (int): stride
    - c_in (int): input channels
    - c_out (int): output channels

    Returns:
    - h_out (int): output height
    - w_out (int): output width
    - c_out (int): output channels (same as input number of filters)
    """
    h_out = (h - k + 2 * p) // s + 1
    w_out = (w - k + 2 * p) // s + 1
    return h_out, w_out, c_out


def max_pooling_dims(h, w, c_in, pool_size, stride):
    """
    Compute the output height, width, and number of channels for a max pooling layer.

    Args:
    - h (int): input height
    - w (int): input width
    - c_in (int): input channels
    - pool_size (int): size of pooling window (usually 2 or 3)
    - stride (int): stride of pooling

    Returns:
    - h_out (int): output height
    - w_out (int): output width
    - c_in (int): output channels (same as input channels)
    """
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    return h_out, w_out, c_in


def flatten_dims(h, w, c_in):
    """
    Flatten the input tensor dimensions into a single dimension.

    Args:
    - h (int): input height
    - w (int): input width
    - c_in (int): input channels

    Returns:
    - flattened_dim (int): the total flattened size (h * w * c_in)
    """
    return h * w * c_in


def fc_dims(dim_in, dim_out):
    """
    Compute the output dimensions of a fully connected layer.

    Args:
    - dim_in (int): input dimension
    - dim_out (int): output dimension

    Returns:
    - dim_out (int): output dimension (unchanged from input)
    """
    return dim_out


def main():

    cnn1_out_dims = cnn_dims(Consts.PICTURE_HEIGHT,Consts.PICTURE_WIDTH,
                             Consts.CONV_2D_KERNEL,Consts.CONV_2D_PADDING,Consts.CONV_2D_STRIDE,Consts.CONV_2D_INPUT_CHANNELS, Consts.CONV_2D_OUTPUT_CHANNELS)

    max1_poolling_out_dims = max_pooling_dims(*cnn1_out_dims,Consts.MAX_POOLING_POOL_SIZE, Consts.MAX_POOLING_STRIDE)
    h1, w1,c_out1 = max1_poolling_out_dims

    cnn2_out_dims = cnn_dims(h1, w1,
                             Consts.CONV_2D_KERNEL, Consts.CONV_2D_PADDING, Consts.CONV_2D_STRIDE,
                             c_out1, c_out1*2)

    max2_poolling_out_dims = max_pooling_dims(*cnn2_out_dims, Consts.MAX_POOLING_POOL_SIZE, Consts.MAX_POOLING_STRIDE)

    flatten_out_dims = flatten_dims(*max2_poolling_out_dims)


    # Define the architecture
    model = Model([
        Conv2D(input_channels=Consts.CONV_2D_INPUT_CHANNELS, output_channels=Consts.CONV_2D_OUTPUT_CHANNELS,
               kernel_size=Consts.CONV_2D_KERNEL, stride=Consts.CONV_2D_STRIDE, padding=Consts.CONV_2D_PADDING),  # Convolution
        BatchNormalization(num_features=Consts.CONV_2D_OUTPUT_CHANNELS),  # Batch Normalization
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2),  # Max Pooling
        Conv2D(input_channels=Consts.CONV_2D_OUTPUT_CHANNELS, output_channels=Consts.CONV_2D_OUTPUT_CHANNELS * 2,
               kernel_size=Consts.CONV_2D_KERNEL, stride=Consts.CONV_2D_STRIDE, padding=Consts.CONV_2D_PADDING),
        BatchNormalization(num_features=c_out1*2),  # Batch Normalization
        # Convolution
        # BatchNormalization(num_features=Consts.CONV_2D_OUTPUT_CHANNELS),  # Batch Normalization
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2),  # Max Pooling
        Flatten(),  # Flatten the output
        FullyConnected(input_tuple=(Consts.BATCH_SIZE, flatten_out_dims), output_size=(Consts.BATCH_SIZE, Consts.NUM_NUIRONS)),  # Fully Connected Layer
        FullyConnected(input_tuple=(Consts.BATCH_SIZE,Consts.NUM_NUIRONS), output_size=(Consts.BATCH_SIZE,Consts.NUM_CLASIFICATION_NUIRONS)),  # Output Layer (e.g., 10 classes for classification)
        Softmax()
    ])

    trained_model, X_validation, Y_validation = Train.train(model)
    # reshape for 32 rows, 32 columns, 3 channels RGB
    X_validation = np.reshape(X_validation, (1000, 32, 32, 3))
    Y_pred = trained_model.inference(X_validation)
    class_accuracies = []
    for cls in range(1,11):
        cls_indices = Y_validation == cls
        cls_correct = np.sum(Y_validation[cls_indices] == Y_pred[cls_indices])
        cls_total = np.sum(cls_indices)
        accuracy = cls_correct / cls_total if cls_total > 0 else 0
        class_accuracies.append(accuracy)
    print(class_accuracies)
    Visualizations.generate_visualizations(Y_validation, Y_pred)
    predict(trained_model)

def predict(model: Model):
    data = np.genfromtxt (Consts.TEST_PATH, delimiter=',')
    X_test = data[:,1:]
    X_test = np.reshape(X_test, (X_test.shape[0], 32, 32, 3))
    predictions = model.forward(X_test)
    test_predictions = np.argmax(predictions, axis=1) + 1
    with open("output.txt",'w') as f:
        for prediction in test_predictions:
            f.write(f"{str(prediction)}\n")

if __name__=="__main__":
    main()