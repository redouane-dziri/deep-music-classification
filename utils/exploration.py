import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import librosa
import librosa.display

import IPython
from git_root import git_root

from utils import load_config, load_params



def check_length(data, length, asserting=True):
    """This function checks that the length of the training data is equal to the
    passed `length`
    
    Arguments:
        data {dict} -- dict containing all of the data
        length {int} -- integer length to compare to
        asserting {boolean} -- whether to throw an error if the lengths don't
            match
    """
    print(f"Actual length of the train data: {len(data['train'])}")
    print(f"Expected length of the train data: {length}")
    if asserting:
        assert len(data['train']) == length      


def extract_from_config(config, params=None):
    """"This function extracts some useful information from the `config` and
    `params` dictionaries
    
    Arguments:
        config {dict} -- a configuration dictionary loaded from `config.json`
        params {dict} -- the feature engineering part of the configuration

    Returns:
        train_root {string} -- the string path to the train data
        genres {list} -- the list of genre names
        n_trains_per_genre {int} -- the number of training examples per genre
        n_pieces {int} -- the number of pieces per track
        n_pieces_per_genre {int} -- the total number of train pieces per genre
        sampling_rate {int} -- the sampling rate for reading the audio data
    """
    data_root = git_root("data", "sample_data")
    train_root = os.path.join(data_root, "train")

    genres = config["genres"]
    total_n_genres = len(genres)
    n_train = config["train_percent"] * config["sample_data_per_genre"] // 10

    n_train_per_genre = n_train // total_n_genres

    n_pieces = params["divide"]["number_pieces"] if params else None

    n_train_pieces = n_train * n_pieces if params else None
    n_pieces_per_genre = n_train_pieces // total_n_genres if params else None

    sampling_rate = params["sampling_rate"] if params else None

    return (train_root, genres, n_train_per_genre, n_pieces, n_pieces_per_genre, 
        sampling_rate)


def display_audio(track_name):
    """This functions displays an audio players in a notebook to
    listen to a track
    
    Arguments:
        track_name {string} -- name of the track
    """
    data_root = git_root("data", "sample_data")
    train_root = os.path.join(data_root, "train")

    print(track_name)
    IPython.display.display(
        IPython.display.Audio(
            os.path.join(train_root, track_name.split(".")[0], track_name)
        )
    )


def display_some_audio(track_names, ask_user=False):
    """This functions iterates through a list of track names and displays an 
    audio player for each one in a notebook. 
    
    Arguments:
        track_names {list} -- list of string names of the tracks
        ask_user {boolean} -- whether to ask users to keep listening after each
            display
    """
    n_tracks = len(track_names)
    for i, track in enumerate(track_names):
        display_audio(track)
        if ask_user:
            if i < n_tracks - 1:
                keep_going = input("Keep listening to more? (Y/N)")
                if keep_going != "Y":
                    break


def display_random_audio(n_genre=3, n_tracks_per_genre=2, seed=11):
    """This functions displays some audio players in a notebook to
    listen to randomly selected tracks from randomly selected genres.
    To listen to tracks from all genres, set `n_genre` to `len(config["genres"]`
    after loading the `config` dict using `utils.load_config()`
    
    Arguments:
        n_genre {int} -- the number of genres to sample is no track list is
            provided
        n_tracks_per_genre {int} -- the number of tracks to sample per genre if
            no track list is provided
        seed {int} -- the random seed for sampling
    """
    config = load_config()
    train_root, genres, n_train_per_genre, _, _, _ = extract_from_config(config)

    R = np.random.RandomState(seed)
    track_genres = R.choice(genres, n_genre, replace=False)
    tracks_indices = R.choice(
        n_train_per_genre, 
        (n_genre, n_tracks_per_genre),
        replace=False
    )
    print(f"Genres picked are: {track_genres} \n")
    
    for g in range(n_genre):
        current_genre = track_genres[g]
        track_list = os.listdir(os.path.join(train_root, current_genre))
        print(f"Tracks for genre - {current_genre}")
        for t in range(n_tracks_per_genre):
            track_name = track_list[tracks_indices[g, t]]
            print(track_name)
            IPython.display.display(
                IPython.display.Audio(
                    os.path.join(train_root, current_genre, track_name)
                )
            )


def print_element(dict, piece_wise=True, i=None, seed=11):
    """This function prints some information from an element in `dict`. If `i`
    is `None` it will randomly sample an element. 

    Arguments:
        dict {dict} -- the dictionary from which to print a train element
        piece_wise {boolean} -- whether the dictionary contains data from pieces
            of tracks or from the full tracks
        i {int} -- if `None` the element shown will be randomly sample; else the
            index of the element to print in the training data in `dict`
        seed {int} -- the seed for random sampling if `i` is `None`
    """
    n = len(dict["train"])

    if not i:
        R = np.random.RandomState(seed)
        i = R.randint(n)

    element = dict['train'][i]

    print(
        f"""Example:
        file_name: {element[0]}
        numpy_representation: {element[1]}
        shape of the array: {element[1].shape}
        {f"piece_number: {element[2]}" if piece_wise else ''}
        genre: {element[-1]}
        """
    )

    return i


def find_short_term_pieces_for(dict, track):
    """"This function retrieves the index at which the track arrays start in
    `dict` (one array if `dict` stores one element per track, several if `dict`
    contains several pieces per track)
    
    Arguments:
        dict {dict} -- a dictionary containing tracks or pieces of tracks
        track {string} -- the name of an audio file
    
    Raises:
        Exception: thrown if the audio file is not registered in `dict`
    
    Returns:
        i {int} -- the index of the first reference to `track` in `dict`
    """
    for i, piece in enumerate(dict["train"]):
        if piece[0] == track:
            return i
    raise Exception("Track not found.")


def display_random_audio_pieces(
    pieces, n_genre=2, n_tracks_per_genre=2, n_short_term=2, seed=11
):
    """This function takes in a dictionary (`pieces`) of audio pieces and 
    displays the audio of randomly selected pieces in `pieces`
    
    Arguments:
        pieces {dict} -- a dictionary containing audio pieces
        n_genre {int} -- the number of genres to randomly select
        n_tracks_per_genre {int} -- the number of tracks to randomly sample in
            each sampled genre
        n_short_term {int} -- the number of pieces of the tracks sampled to
            randomly sample
        seed {int} -- the random seed used for sampling
    """

    R = np.random.RandomState(seed)

    config = load_config()
    params = load_params()
    train_root, genres, n_train_per_genre, _, _, sampling_rate = extract_from_config(
        config, params
    )
    
    random_genres = R.choice(genres, n_genre, replace=False)
    random_tracks = R.choice(
        n_train_per_genre, 
        (n_genre, n_tracks_per_genre),
        replace=False
    )
    random_pieces = R.choice(
        params["divide"]["number_pieces"],
        (n_genre, n_tracks_per_genre, n_short_term),
        replace=False
    )
    
    print(f"Genres picked are: {random_genres} \n")
    
    for g in range(n_genre):
        current_genre = random_genres[g]
        track_list = os.listdir(os.path.join(train_root, current_genre))
        print(f" -- Tracks for genre - {current_genre} -- \n ")
        for t in range(n_tracks_per_genre):
            track = track_list[random_tracks[g, t]]
            print("Full track:")
            print(track)
            IPython.display.display(
                IPython.display.Audio(
                    os.path.join(train_root, current_genre, track)
                )
            )
            start_index = find_short_term_pieces_for(pieces, track)
            print(start_index)
            print("Random short-term pieces of the track:")
            for index in random_pieces[g, t]:
                print(f"piece {index}")
                IPython.display.display(
                    IPython.display.Audio(
                        pieces["train"][start_index + index][1],
                        rate = sampling_rate
                    )
                )


def plot_one_map(data_map, map_type, quantized=False):
    """This function plots one map (either a spectrogram or a mel map)

    Arguments:
        data_map {tuple} -- a tuple of the form (file_name, numpy_map, piece_id,
            genre)
        map_type {string} -- one of ('spectrogram', 'mel_map') for the type of
            map to print (affects the y-scale and text)
        quantized {boolean} -- whether the input maps have already been
            quantized or not (will handle the color scale)
    """

    params = load_params()

    frame_length = int(
        params[map_type]["frame_length_in_s"] * params["sampling_rate"]
    )
    hop_length = int((1 - params[map_type]["overlap"]) * frame_length)

    plt.title(f"""
    ---------------- {map_type.capitalize()} ----------------
    ------- Genre - {data_map[-1]} ------- 
    """
    )
    y_axis = "log" if map_type == "spectrogram" else "mel"
    librosa.display.specshow(
        data_map[1], 
        sr=params["sampling_rate"], 
        hop_length=hop_length, 
        x_axis="time", 
        y_axis=y_axis
    )
    
    colorbar_format = "%i" if quantized else "%+2.0f dB"
    
    plt.colorbar(format=colorbar_format) 


def plot_random_maps(maps, n_per_genre, map_type, seed=11):
    """This function prints some random maps from a `maps` dictionary for each
    genre 

    Arguments:
        maps {dict} -- a dictionary containing the tuples of maps
        n_per_genre {int} -- the number of maps to plot per genre
        map_type {string} -- one of ('spectrogram', 'mel_map') for the type of
            map to print (affects the y-scale and text)
        seed {int} -- the random seed used for sampling
    """

    R = np.random.RandomState(seed)

    config = load_config()
    params = load_params()
    _, genres, _, _, n_pieces_per_genre, sampling_rate = extract_from_config(
        config, params
    )

    n_genres = len(genres)
    
    random_tracks = R.randint(
        0, n_pieces_per_genre, (n_genres, n_per_genre)
    )
    
    frame_length = int(
        params[map_type]["frame_length_in_s"] * sampling_rate
    )
    hop_length = int(
        (1 - params[map_type]["overlap"]) * frame_length
    )
    
    fig = plt.figure(figsize = (2 * n_genres, int(2.5 * n_genres)))
    subplot = 0
    for g in range(n_genres):
        for t in range(n_per_genre):
            subplot += 1
            map_index = int(g * n_pieces_per_genre + random_tracks[g, t])
            data_to_plot = maps["train"][map_index]
            ax = fig.add_subplot(n_genres, n_per_genre, subplot)
            ax.set_title(f"-- {data_to_plot[-1]} --")
            y_axis = "log" if map_type == "spectrogram" else "mel"
            librosa.display.specshow(
                data_to_plot[1], 
                sr=sampling_rate, 
                hop_length=hop_length, 
                x_axis="time", 
                y_axis=y_axis
            )
            ax.axis("off")
            plt.tight_layout()   

    plt.show()   


def display_unique_extremes(maps, map_type):
    """This function displays the unique minima and maxima of some maps and
    returns the unique minima and maxima for other functions that might need
    them

    Arguments:
        maps {dict} -- a dictionary containing the tuples of maps
        map_type {string} -- one of ('spectrogram', 'mel_map')
    
    Returns:
        {np.array} -- float array containing the unique minima
        {np.array} -- float array containing the unique maxima
    """

    n_pieces = len(maps["train"])
    
    mins = np.array([maps["train"][i][1].min() for i in range(n_pieces)])
    maxs = np.array([maps["train"][i][1].max() for i in range(n_pieces)])
    
    print(f"""Unique mins of {map_type} are:
{np.unique(mins)} \n
Unique maxs of {map_type} are:
{np.unique(maxs)}
    """
    )

    return np.unique(mins), np.unique(maxs)


def display_genre_min_counts(quantized_maps):
    """This function displays bar charts faceted by genres for the value of the
    minimum of the quantized_maps
    
    Arguments:
        quantized_maps {dict} -- dictionary containing the tuples of quantized
            maps
    """
    config = load_config()

    n_pieces = len(quantized_maps["train"])

    mins = np.array([
        quantized_maps["train"][i][1].min() for i in range(n_pieces)
    ])
    unique_mins = np.unique(mins)

    genre_min_count = {
        genre: np.zeros(len(unique_mins)) for genre in config["genres"]
    }

    n_pieces = len(quantized_maps["train"])
    for i in range(n_pieces):
        piece = quantized_maps["train"][i]
        genre_min_count[piece[-1]][piece[1].min() - 1] += 1
    
    min_count_df = pd.DataFrame(genre_min_count)
    min_count_df = pd.melt(min_count_df, value_vars = min_count_df.columns)
    min_count_df["min"] = min_count_df.groupby("variable").cumcount() + 1

    plt.figure(figsize = (15, 7))
    for i, genre in enumerate(min_count_df["variable"].unique()):
        min_count_genre = min_count_df[min_count_df["variable"] == genre]
        ax = plt.subplot(2, 5, i + 1)
        ax.set_title(genre)
        ax.bar(
            min_count_genre["min"].astype(str), 
            min_count_genre["value"]
        )
    plt.show()


def plot_one_glcm(glcm, levels):
    """This function plots one GLCM

    Arguments:
        glcm {tuple} -- a tuple of the form (file_name, numpy_glcm, piece_id,
            genre)
    """

    plt.imshow(glcm[1], cmap="gray", interpolation="nearest")
    plt.title(f"""
    ---------------- GLCM ----------------
    ------- Genre - {glcm[-1]} ------- 
    ------- Track - {glcm[0]} -------
    """
    )
    plt.xticks(np.arange(0, levels))
    plt.yticks(np.arange(0, levels))
    plt.show()


def plot_random_GLCMs(glcms, n_per_genre, seed=11):
    """This function prints some random GLCMs from a `glcms` dictionary for each
    genre 

    Arguments:
        glcms {dict} -- a dictionary containing the tuples of glcms
        n_per_genre {int} -- the number of glcms to plot per genre
        seed {int} -- the random seed used for sampling
    """

    R = np.random.RandomState(seed)

    config = load_config()
    params = load_params()
    _, genres, _, _, n_pieces_per_genre, _ = extract_from_config(
        config, params
    )

    n_genres = len(genres)
    
    random_tracks = R.randint(
        0, n_pieces_per_genre, (n_genres, n_per_genre)
    )
    
    fig = plt.figure(figsize = (int(1.25 * n_genres), int(2.5 * n_genres)))
    subplot = 0
    for g in range(n_genres):
        for t in range(n_per_genre):
            subplot += 1
            map_index = int(g * n_pieces_per_genre + random_tracks[g, t])
            data_to_plot = glcms["train"][map_index]
            ax = fig.add_subplot(n_genres, n_per_genre, subplot)
            ax.set_title(f"-- {data_to_plot[-1]} --")
            plt.imshow(data_to_plot[1], cmap="gray", interpolation="nearest")
            ax.axis("off")
            plt.tight_layout()   

    plt.show()   


def plot_one_mfcc(mfcc, time_id):
    """This function plots a MFCC 
    
    Arguments:
        mfcc {tuple} -- tuple of the form ('file_name', 'numpy_mfcc', 'genre')
            where `numpy_mfcc` is of shape 
            (n_submaps, n_mfcc, n_windows/n_submaps)
        time_id {integer} -- the submap to plot
    """

    params = load_params()

    plt.title(f"MFCC for {mfcc[0]}")
    librosa.display.specshow(
        mfcc[1][time_id], sr=params["sampling_rate"], x_axis="time"
    )
    plt.colorbar()

    plt.show()


def plot_random_mfccs(mfccs, n_per_genre, seed=11):
    """This function prints some random MFCCs from a `mfccs` dictionary for each
    genre 

    Arguments:
        mfccs {dict} -- a dictionary containing the tuples of mfccs
        n_per_genre {int} -- the number of mfccs to plot per genre
        seed {int} -- the random seed used for sampling
    """

    R = np.random.RandomState(seed)

    config = load_config()
    params = load_params()
    _, genres, n_train_per_genre, _, _, sampling_rate = extract_from_config(
        config, params
    )

    n_genres = len(genres)
    
    random_tracks = R.randint(
        0, n_train_per_genre, (n_genres, n_per_genre)
    )
    
    fig = plt.figure(figsize = (2* n_genres, int(2.5 * n_genres)))
    subplot = 0
    for g in range(n_genres):
        for t in range(n_per_genre):
            submap = R.randint(0, params["MFCC"]["n_submaps"])
            subplot += 1
            map_index = int(g * n_train_per_genre + random_tracks[g, t])
            data_to_plot = mfccs["train"][map_index]
            ax = fig.add_subplot(n_genres, n_per_genre, subplot)
            ax.set_title(f"-- {data_to_plot[-1]} --")
            librosa.display.specshow(
                data_to_plot[1][submap], sr=sampling_rate, x_axis="time"
            )
            ax.axis("off")
            plt.tight_layout()   

    plt.show()  
