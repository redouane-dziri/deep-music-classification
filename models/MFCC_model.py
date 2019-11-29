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
    input_dim_1 = params["MFCC"]["n_mfcc"]
    #input_dim_3 = params["MFCC"]["n_submaps"]
    input_dim_3 = 1 #piece-wise model (there are 30 submaps per track)
    input_dim_2 = params["MFCC"]["n_windows"] // params["MFCC"]["n_submaps"] #time length of a submap
    n_genres = len(config["genres"])

    model = Sequential()
    model.add(
        Conv2D(
            12, 
            (10, 10), 
            activation="tanh", 
            input_shape=(input_dim_1, input_dim_2, input_dim_3),
            name="conv1"
        )
    )
    model.add(AveragePooling2D((4, 4), name="pooling1"))
    model.add(AffineScalar(name="affine"))
    model.add(Conv2D(12, (3,3), activation="tanh", name="conv2"))
    model.add(GlobalAveragePooling2D(name="pooling2"))
    #model.add(AveragePooling2D((4, 4), name="pooling1"))
    #model.add(AffineScalar(name="affine"))
    model.add(Dense(n_genres, activation="softmax", name="dense"))

    return model
