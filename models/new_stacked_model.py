import sys

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, LeakyReLU

from git_root import git_root

#Load the utils module
sys.path.append(git_root("utils"))
from utils import load_config, load_params



def setup_model_dense(M=4):

    config = load_config()
    params = load_params()
    input_dim = params["quantization"]["n_levels"] - 1
    n_genres = len(config["genres"])

    model = Sequential()
    model.add(
        Conv2D(
            24, 
            (3, 3),
            input_shape=(input_dim, input_dim, M),
            name="conv1"
        )
    )
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2, 2), name="pooling1"))
    model.add(Conv2D(12, (2, 2), name="conv2"))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(32, name="dense1"))
    model.add(LeakyReLU())
    model.add(Dense(n_genres, activation="softmax", name="dense2"))

    return model
    