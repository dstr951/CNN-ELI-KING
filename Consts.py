from Optimizer import Optimizer
PICTURE_WIDTH = 32
PICTURE_HEIGHT = 32
TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validate.csv"
TEST_PATH = "data/test.csv"
RESULT_PATH = ""
RESULT_NAME = ""
SEED = 42
TRAIN_NUM = 0
BATCH_SIZE = 50
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

#Optimizer params
OPTIMIZER_METHOD = Optimizer.METHOD_ADAM


#CNN HYPER PARAMETERS
CONV_2D_INPUT_CHANNELS = 3
CONV_2D_OUTPUT_CHANNELS = 8
CONV_2D_KERNEL = 3
CONV_2D_STRIDE = 1
CONV_2D_PADDING = 1

# max pooling HYPER PARAMETERS
MAX_POOLING_POOL_SIZE = 2
MAX_POOLING_STRIDE = 2


# FULLY CONNECTED HYPER PARAMETERS
NUM_NUIRONS = 128


#
NUM_CLASIFICATION_NUIRONS = 10