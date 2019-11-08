from git_root import git_root
import os
import sys
import json

import numpy as np

from generate_melmap import generate_mel_map
from generate_spectrogram import generate_spectrogram
from generate_GLCM import generate_glcm


# <---- For importing a .py file from another package ---->
sys.path.append(os.path.join(git_root(),'utils'))
sys.path.append(os.path.join(git_root(),'feature_engineering'))
from utils import read_in_data, generate_short_term_piece, quantize, load_params, load_config


def generate_short_term_pieces_from_dict(data):
    """This function generates a the short term pieces for all tracks of the
    dataset
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'genre')

    Returns:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'split_id', 'genre')
    """

    # track[0] is the "file_name"
    # piece[0] is the numpy representation of the chunk
    # piece[1] is the split id
    # track[2] is the "genre"

    params = load_params()

    for split in data:
	    data[split] = [
            (track[0], piece[0], piece[1], track[2]) 
                for track in data[split]
                for piece in generate_short_term_piece(
				    track[1],
				    number_pieces=params["divide"]["number_pieces"], 
				    sampling_rate=params["sampling_rate"],
				    piece_length=params["divide"]["piece_length_in_s"],
				    overlap=params["divide"]["overlap_in_s"]
                ) 
        ]
    
    return data


def generate_mel_maps_from_dict(data):
    """This function generates a mel map for all short time pieces of the
    dataset
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'split_id', 'genre')

    Returns:
        mel_maps {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'mel_map', 'split_id', 'genre')
    """

    params = load_params()

    mel_maps = {"train": None, "test": None}
    hop_length = int(
        params["mel_map"]["hop_length_in_s"] * params["sampling_rate"]
    )

    for split in mel_maps:
	    mel_maps[split] = [
            (
                piece[0], 
                generate_mel_map(
                    piece[1], 
                    sampling_rate=params["sampling_rate"],
		            hop_length=hop_length,
		            n_mels=params["mel_map"]["n_mels"]
                ), 
                piece[2], 
                piece[3]
            ) for piece in data[split]
        ]
    
    return mel_maps


def generate_spectrograms_from_dict(data):
    """This function generates a spectrogram for all short term pieces of the
    dataset
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'split_id', 'genre')

    Returns:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'spectrogram', 'split_id', 'genre')
    """

    params = load_params()

    spectrograms = {"train": None, "test": None}
    hop_length = int(
        params["spectrogram"]["hop_length_in_s"] * params["sampling_rate"]
    )

    for split in spectrograms:
	    spectrograms[split] = [
            (
                piece[0], 
                generate_spectrogram(piece[1], hop_length=hop_length), 
                piece[2], 
                piece[3]
            ) for piece in data[split]
        ]
    
    return spectrograms


def generate_quantized_maps_from_dict(maps):
    """This function quantizes the maps
    
    Arguments:
        maps {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_map', 'split_id', 'genre')

    Returns:
        quantized_maps {dict} -- keys in ('train', 'test'), values are lists 
        of tuples ('file_name', 'quantized_map', 'split_id', 'genre')
    """

    params = load_params()

    quantized_maps = {"train": None, "test": None}

    for split in quantized_maps:
	    quantized_maps[split] = [
            (
                piece[0], 
                quantize(
                    piece[1], n_levels=params["quantization"]["n_levels"]
                ), 
                piece[2], 
                piece[3]
            ) for piece in maps[split]
        ]
    
    return quantized_maps


def generate_glcms_from_dict(maps, map_type="mel_map"):
    """This function generates GLCMs for all maps with co-occurrence computed
    between pairs with the distance and angles in the configuration json
    
    Arguments:
        maps {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_map', 'split_id', 'genre')
        map_type {string} -- one of ('mel_map', 'spectrogram')

    Returns:
        glcms {list} -- list of dicts with keys in ('train', 'test'), and values
            lists of tuples ('file_name', 'glcm', 'split_id', 'genre'),
            one dict per angle
    """

    params = load_params()

    angles_in_deg = params["GLCM"][map_type]["angles_in_deg"]
    glcms = [{"train": None, "test": None}]*len(angles_in_deg)

    for i, angle in enumerate(angles_in_deg):
        for split in glcms[i]:
	        glcms[i][split] = [
                (
                    piece[0], 
                    generate_glcm(
                        piece[1], 
                        distance=params["GLCM"]["distance"],
                        angle_in_deg=angle
                    ), 
                    piece[2], 
                    piece[3]
                ) for piece in maps[split]
            ]
    
    return glcms


def preprocess_data():
    """This function reads in the data, computes the successive maps needed and
    outputs a list of dicts containing the appropriate GLCM to feed in the
    neural network

    Returns:
        glcms {dict} -- keys k are 'spectrogram', 'mel_map'
            glcms[k] {list}  -- each element corresponds to a different angle
                amongst angles listed in `config.json`
                glcms[k][i] {dict} -- keys s are 'train', 'test'
                    glcms[k][i][s] {list} -- each element is a tuple of
                        ('file_name', 'glcm', 'split_id', 'genre') and
                        corresponds to the GLCM built from the maps extracted
                        for a short-term piece in a file with the associated
                        name and genre
    """
    # STEP 1: Load the configurations
    # ------------------------------------------------------------------------------
    params = load_params()
    config = load_config()
    
    # STEP 2: Load the data
    # --------------------------------------------------------------------------

    # read .wav files into np arrays
    data = read_in_data(
        params["sampling_rate"], sample_data=config["using_sample_data"]
    )

    # STEP 3: Cut the data into smaller chunks
    # --------------------------------------------------------------------------

    data = generate_short_term_pieces_from_dict(data)

    # STEP 4: Generate mel maps for each chunk
    # --------------------------------------------------------------------------

    # convert the numpy representations of .wav files to mel_maps
    mel_maps = generate_mel_maps_from_dict(data)

    # STEP 5: Generate spectrograms for each chunk
    # --------------------------------------------------------------------------

    # convert the numpy representations of .wav files to spectrograms
    spectrograms = generate_spectrograms_from_dict(data)

    # STEP 6: Quantize the maps
    # --------------------------------------------------------------------------

    quantized_mel_maps = generate_quantized_maps_from_dict(mel_maps)
    quantized_spectrograms = generate_quantized_maps_from_dict(spectrograms)

    # STEP 7: Generate the GLCMs
    # --------------------------------------------------------------------------

    glcms_from_spectrogram = generate_glcms_from_dict(
        quantized_spectrograms, map_type="spectrogram"
    )

    glcms_from_mel_map = generate_glcms_from_dict(
        quantized_mel_maps, map_type="mel_map"
    )

    glcms = {
        "spectrogram": glcms_from_spectrogram, "mel_map": glcms_from_mel_map
    }
    
    return glcms

if __name__=="__main__":

    x = preprocess_data()

    #### CURRENTLY: ISSUES TO SERIALIZE ONE OF THE NP ARRAys

    for key, value in x.items():
        for j in range(len(value)):
            for key2, value2 in value[j].items():
                for i in range(len(value2)):
                    try:
                        test = value2[i][1].tolist()
                    except:
                        print("Problem")
                        if(isinstance(value2[i][1], (list))):
                            print("Problem")
                            print(key)
                            print(j)
                            print(key2)
                    

    with open(os.path.join(git_root(),'data','pipeline_output','test.json'),'w') as outfile:
        json.dump(x, outfile)