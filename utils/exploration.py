import os

import numpy as np
import matplotlib.pyplot as plt

import librosa

import IPython
from git_root import git_root

from utils import load_config, load_params



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

    n_train_pieces = n_train*params["divide"]["number_pieces"] if params else None
    n_pieces_per_genre = n_train_pieces // total_n_genres if params else None

    sampling_rate = params["sampling_rate"] if params else None

    return train_root, genres, n_train_per_genre, n_pieces, n_pieces_per_genre, sampling_rate


def display_random_audio(n_genre=3, n_tracks_per_genre=2, seed=11):
    """This functions displays some audio players in a notebook to
    listen to randomly selected tracks from randomly selected genres.
    To listen to tracks from all genres, set `n_genre` to `len(config["genres"]`
    after loading the `config` dict using `utils.load_config()`
    
    Arguments:
        n_genre {int} -- the number of genres to sample
        n_tracks_per_genre {int} -- the number of tracks to sample per genre
        seed {int} -- the random seed for sampling
    """

    R = np.random.RandomState(seed)

    config = load_config()
    train_root, genres, n_train_per_genre, _, _, _ = extract_from_config(config)

    random_genres = R.choice(genres, n_genre, replace=False)
    random_tracks = R.choice(
        n_train_per_genre, 
        (n_genre, n_tracks_per_genre),
        replace=False
    )
    print(f"Genres picked are: {random_genres} \n")
    print(f"Track indices for each genres are: {random_tracks} \n")
    
    for g in range(n_genre):
        current_genre = random_genres[g]
        track_list = os.listdir(os.path.join(train_root, current_genre))
        print(f"Tracks for genre - {current_genre}")
        for t in range(n_tracks_per_genre):
            IPython.display.display(
                IPython.display.Audio(
                    os.path.join(
                        train_root, 
                        current_genre, 
                        track_list[random_tracks[g, t]]
                    )
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
    train_root, genres, n_train_per_genre, _, n_pieces_per_genre, _ = extract_from_config(
        config, params
    )
    
    random_genres = R.choice(genres, n_genre, replace=False)
    random_tracks = R.choice(
        n_train_per_genre, 
        (n_genre, n_tracks_per_genre),
        replace=False
    )
    random_pieces = R.choice(
        n_pieces_per_genre,
        (n_genre, n_tracks_per_genre, n_short_term),
        replace=False
    )
    
    print(f"Genres picked are: {random_genres} \n")
    print(f"Track indices for each genres are: {random_tracks} \n")
    print(f"Short pieces indices for each tracks are: {random_pieces} \n")
    
    for g in range(n_genre):
        current_genre = random_genres[g]
        track_list = os.listdir(os.path.join(train_root, current_genre))
        print(f" -- Tracks for genre - {current_genre} -- \n ")
        for t in range(n_tracks_per_genre):
            track = track_list[random_tracks[g, t]]
            print("Full track:")
            IPython.display.display(
                IPython.display.Audio(
                    os.path.join(train_root, current_genre, track)
                )
            )
            start_index = find_short_term_pieces_for(pieces, track)
            print("Random short-term pieces of the track:")
            for index in random_pieces[g][t]:
                IPython.display.display(
                    IPython.display.Audio(
                        pieces["train"][start_index + index][1],
                        rate = params["sampling_rate"]
                    )
                )


def plot_one_map(data_map, map_type):
    """This function plots one map (either a spectrogram or a mel map)

    Arguments:
        data_map {tuple} -- a tuple of the form (file_name, numpy_map, piece_id,
            genre)
        map_type {string} -- one of ('spectrogram', 'mel_map') for the type of
            map to print (affects the y-scale and text)
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
    
    plt.colorbar(format='%+2.0f dB') 


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
