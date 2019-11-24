import sys

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, Dense

from git_root import git_root

#Load the utils module
sys.path.append(git_root("utils"))
from utils import load_config, load_params
from affine_scalar_layer import AffineScalar



def setup_model():
    """[summary]
    
    Returns:
        [type] -- [description]
    """

    config = load_config()
    params = load_params()
    input_dim = params["quantization"]["n_levels"] - 1
    n_genres = len(config["genres"])

    model = Sequential()
    model.add(
        Conv2D(
            12, 
            (6, 6), 
            activation="tanh", 
            input_shape=(input_dim, input_dim, 1)
        )
    )
    model.add(AveragePooling2D((2, 2)))
    model.add(AffineScalar())
    model.add(Conv2D(6, (3,3), activation="tanh"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_genres, activation="softmax"))

    return model
