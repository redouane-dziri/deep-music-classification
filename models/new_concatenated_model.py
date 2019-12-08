import sys

from tensorflow import keras
from tensorflow.keras import layers

from git_root import git_root

#Load the utils module
sys.path.append(git_root("utils"))
from utils import load_config, load_params
from affine_scalar_layer import AffineScalar



def concatenated_model():

    config = load_config()
    params = load_params()
    input_dim = params["quantization"]["n_levels"] - 1
    n_genres = len(config["genres"])

    inputs_0 = keras.Input(shape=(input_dim, input_dim, 1))
    inputs_45 = keras.Input(shape=(input_dim, input_dim, 1))
    inputs_90 = keras.Input(shape=(input_dim, input_dim, 1))
    inputs_135 = keras.Input(shape=(input_dim, input_dim, 1))

    x0 = layers.Conv2D(
        12, (6, 6), activation="tanh", name="conv1_0"
    )(inputs_0)
    x0 = layers.AveragePooling2D((2, 2), name="pooling1_0")(x0)
    x0 = AffineScalar(name="affine_0")(x0)
    x0 = layers.Conv2D(6, (3,3), activation="tanh", name="conv2_0")(x0)
    x0 = layers.GlobalAveragePooling2D(name="pooling2_0")(x0)

    x45 = layers.Conv2D(
        12, (6, 6), activation="tanh", name="conv1_45"
    )(inputs_45)
    x45 = layers.AveragePooling2D((2, 2), name="pooling1_45")(x45)
    x45 = AffineScalar(name="affine_45")(x45)
    x45 = layers.Conv2D(6, (3,3), activation="tanh", name="conv2_45")(x45)
    x45 = layers.GlobalAveragePooling2D(name="pooling2_45")(x45)

    x90 = layers.Conv2D(
        12, (6, 6), activation="tanh", name="conv1_90"
    )(inputs_90)
    x90 = layers.AveragePooling2D((2, 2), name="pooling1_90")(x90)
    x90 = AffineScalar(name="affine_90")(x90)
    x90 = layers.Conv2D(6, (3,3), activation="tanh", name="conv2_90")(x90)
    x90 = layers.GlobalAveragePooling2D(name="pooling2_90")(x90)

    x135 = layers.Conv2D(
        12, (6, 6), activation="tanh", name="conv1_135"
    )(inputs_135)
    x135 = layers.AveragePooling2D((2, 2), name="pooling1_135")(x135)
    x135 = AffineScalar(name="affine_135")(x135)
    x135 = layers.Conv2D(6, (3,3), activation="tanh", name="conv2_135")(x135)
    x135 = layers.GlobalAveragePooling2D(name="pooling2_135")(x135)

    concatenated = layers.concatenate([x0, x45, x90, x135])
    out = layers.Dense(n_genres, activation="softmax", name="dense")(
        concatenated
    )

    model = keras.Model(
        inputs=[inputs_0, inputs_45, inputs_90, inputs_135], outputs=out
    )

    return model