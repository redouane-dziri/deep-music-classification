import os
import sys

import numpy as np
import pandas as pd

from google.cloud import storage
from git_root import git_root

#Load the utils module
sys.path.append(git_root("utils"))
from utils import load_config



def fetch_data_cloud(map_type, angle, train=True):
    """This function to fetches data from json files in google cloud storage and 
    returns a pandas DataFrame

    Arguments:
        map_type {string} -- one of ('spectrogram', 'mel_map', 'mfcc'), the type
            of map the GLCM was made of or the MFCC map `map`
        angle {string} -- the angle of the GLCM to fetch, in string form (e.g. 
            '0')
        train {boolean} -- whether to fetch the training or testing data

    Returns:
        df {pd DataFrame} -- if `map_type = 'mfcc'`, columns are 
            ['filename', 'map', 'genre'] else 
            ['filename', 'map', 'split_id', 'genre']
            where 'filename' is the name of the track, 'map' is the numeric
            numpy array representation, 'genre' is the track genre and 
            'split_id' is the id of the short-term piece the map was built on
    """

    ### Read data from Google cloud storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaud.stiegler/Desktop/Divers/adl-hw4-675afec62d41.json"
    storage_client = storage.Client("Music-Genre-Classification")
    bucket = storage_client.get_bucket("deep-music-classification")

    if train:
        set_type = "train"
    else:
        set_type = "test"

    if(map_type == "mfcc"):
        blob_name = os.path.join(
            "data", 
            "preprocessed_data",
            map_type,
            "data_{}_{}.json".format(map_type, set_type)
        )
    else:
        blob_name = os.path.join(
            "data",
            "preprocessed_data",
            map_type,
            "data_{}_angle_{}_{}.json".format(map_type, angle, set_type)
        )

    print("Fetching: {}".format(blob_name))

    blob = bucket.get_blob(blob_name)
    content = blob.download_as_string()
    df = pd.read_json(content)  

    if(map_type == "mfcc"):
        df.columns = ["filename", "map", "genre"]
    else:
        df.columns = ["filename", "map", "split_id", "genre"]

    return df


def fetch_data_local(map_type, angle, train=True):
    """
    This function to fetches data from json files locally and returns a pandas 
    DataFrame

    Arguments:
        map_type {string} -- one of ('spectrogram', 'mel_map', 'mfcc'), the type
            of map the GLCM was made of or the MFCC map `map`
        angle {string} -- the angle of the GLCM to fetch, in string form (e.g. 
            '0')
        train {boolean} -- whether to fetch the training or testing data

    Returns:
        df {pd DataFrame} -- if `map_type = 'mfcc'`, columns are 
            ['filename', 'map', 'genre'] else 
            ['filename', 'map', 'split_id', 'genre']
            where 'filename' is the name of the track, 'map' is the numeric
            numpy array representation, 'genre' is the track genre and 
            'split_id' is the id of the short-term piece the map was built on
    """

    if train:
        set_type = "train"
    else:
        set_type = "test"

    if(map_type == "mfcc"):
        file_name = git_root(
            "data",
            "preprocessed_data_full",
            map_type,
            "data_{}_{}.json".format(map_type, set_type)
        )
    else:
        file_name = git_root(
            "data",
            "preprocessed_data_full",
            map_type,
            "data_{}_angle_{}_{}.json".format(map_type, angle, set_type)
        )

    print("Fetching: {}".format(file_name))

    df = pd.read_json(file_name)  

    if(map_type == "mfcc"):
        df.columns = ["filename", "maps", "genre"]
    else:
        df.columns = ["filename", "maps", "split_id", "genre"]

    return df


def to_numpy_arrays(df):
    """This function takes in a dataframe with columns 
    []'filename', 'maps', 'genre'] like one output by `fetch_data_local` or
    `fetch_data_cloud` and returns two np arrays `samples` and `labels`
    containing, respectively, the numpy maps and their associated labels
    
    Arguments:
        df {pd DataFrame} -- with columns ['filename', 'map', 'genre'] where 
            'filename' is the name of the track, 'map' is the numeric numpy 
            array representation, 'genre' is the track genre 
    
    Returns:
        samples, labels {(list, list)} -- list of numpy arrays containing the
            maps and their associated labels
    """
    config = load_config()
    label_names = config["genres"]
    label_to_idx = dict((name, index) for index, name in enumerate(label_names))

    samples = []
    labels = []
    for _, row in df.iterrows():
        samples.append(np.array(row['maps']))
        labels.append(label_to_idx[row['genre']])
    
    return samples, labels


def prepare_tf_dataset(samples, labels):
    """
    
    Arguments:
        samples {[type]} -- [description]
        labels {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    BATCH_SIZE = 32

    dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
    dataset = dataset.shuffle(128).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset



if __name__ == "__main__":
    print("Fetching from Google Storage")
    df = fetch_data_cloud("spectrogram", angle="0", train=False)
    print(df.shape)
    print("\n")
    print("Fetching from local files")
    df = fetch_data_local("spectrogram", angle="0", train=False)
    print(df.shape)
