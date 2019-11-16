import os
import sys
import io

import numpy as np
import pandas as pd
import json

import librosa

from git_root import git_root



def load_params():
    """Helper function to read the parameters from the config file

    Returns:
        data {dict} containing whatever parameters are in the config file
    """
    with open(git_root("config", "config.json"), "r") as config:
    	    config = json.load(config)
    		
    params = config["feature_engineering"]
    return params


def load_config():
    """Helper function to read the entire config file

    Returns:
        data {dict} containing the entire content of the config file
    """
    with open(git_root("config", "config.json"), "r") as config:
    	    config = json.load(config)
    		
    return config


def read_in_data(sampling_rate, sample_data=True):
    """Function to load the data in memory 

    Arguments:
        sampling_rate {int} -- the sampling rate with which to read the .wav
        sample_data {boolean} -- if True reads data from `sample_data` subfolder
            else reads data from `full_data` subfolder
    Returns:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'genre')
    """
    
    data_root = git_root("data")

    metadata = pd.read_csv(
        os.path.join(data_root, "metadata", "train_test_split.csv")
    )

    train_metadata = metadata.loc[metadata["split" ]== "train", ]
    test_metadata = metadata.loc[metadata["split"] == "test", ]
    if sample_data:
        train_metadata = train_metadata.loc[train_metadata["sample"], ]
        test_metadata = test_metadata.loc[test_metadata["sample"], ]
	
    train_records = []
    test_records = []

    def load_file(metadata_row):
        data_folder = "sample_data" if sample_data else "full_data"
        file_path = os.path.join(
            data_root,
            data_folder,
            metadata_row["split"],
            metadata_row["genre"],
            metadata_row["file_name"]
        )
        file_numpy_representation, _ = librosa.load(file_path, sr=sampling_rate)
        return (
            metadata_row["file_name"], 
            file_numpy_representation, 
            metadata_row["genre"]
        )

    train_records = train_metadata.apply(load_file, axis=1).tolist()
    test_records = test_metadata.apply(load_file, axis=1).tolist()

    return {'train': train_records, 'test': test_records}


def generate_short_term_piece(
    file_array, number_pieces, sampling_rate, piece_length, overlap
):
	"""Function to divide each music piece into short-term pieces (see paper)
    
    Arguments:
        file_array {np.array} -- a floating number numpy array
        number_pieces {int} -- the number of pieces to return
        sampling_rate {int} -- the sampling rate used to generate `file_array` 
            from a .wav file (number of frames per second)
        piece_length {float} -- the length of each piece in seconds
        overlap {float} -- the overlap of the pieces in seconds
    
    Returns:
        view {list} -- a list containing `number_splits` splits tuples of 
            `file_array` with or without overlap, with the first element of size 
            `piece_length * sampling_rate` and the second a split id ranging
            from 0 to `number_pieces`
    """

    # the number of frames for one short-term piece
	frame_length = int(sampling_rate * piece_length)
    # the number of frames of overlap
	frame_overlap = int(overlap * sampling_rate)

	sh = (file_array.size - frame_length + 1, frame_length)
	st = file_array.strides * 2
	view = np.lib.stride_tricks.as_strided(
        file_array, strides = st, shape = sh
    )[0::frame_overlap]
	
	return list(zip(view, range(number_pieces)))


def quantize(array, n_levels):
    """This function quantizes a float array into an int array mapping original 
    decibel values (contained in [-80, 0] in `n_levels` bins linearly spaced in
    `[-80, 0]` (e.g. for `n_levels=16`, will bin in buckets of 5dB)
    
    Arguments:
        array {np.array} -- float array
        n_levels {int} -- the number of bins

    Returns:
        {np.array} -- an int array with the bin index in which each original
            value falls (includes lower boundary of the bins and excludes the
            upper boundary)
    """
    epsilon = 1e-4

    # making sure to include 0dB in the last bin and not its separate bin
    bin_limits = np.linspace(-80, 0, num=n_levels+1)
    bin_limits[-1] = epsilon

    # levels should start at `0` for the `GLCM` downstream calculation
    return np.digitize(array, bin_limits).astype(np.uint8) - 1
    