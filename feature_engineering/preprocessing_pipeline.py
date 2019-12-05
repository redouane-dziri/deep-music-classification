from git_root import git_root
import os
import sys
import json

import numpy as np

from generate_melmap import generate_mel_map
from generate_spectrogram import generate_spectrogram
from generate_GLCM import generate_glcm
from generate_MFCC import generate_MFCC

# <---- For importing a .py file from another module ---->
sys.path.append(os.path.join(git_root(), "utils"))
from utils import read_in_data, generate_short_term_piece
from utils import quantize, load_params, load_config


def pad_from_dict(data):
    """This function pads tracks that are too short to produce 14 chunks in the
    later short term pieces generation
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'genre')

    Returns:
        data_padded {dict} -- keys in ('train', 'test'), values are lists of 
            tuples ('file_name', 'numpy_representation', 'genre')
            and the numpy representations are padded with zeros
    """

    # track[0] is the "file_name"
    # track[1] is the numpy representation of the track
    # track[2] is the "genre"

    params = load_params()

    min_length = params["divide"]["min_piece_length"]

    # important to not overwrite `data`!
    data_padded = {"train": None, "test": None}
    for split in data:
        data_padded[split] = [
            (
                track[0], 
                np.pad(track[1], max(min_length - len(track[1]), 0)),
                track[2]
            ) for track in data[split]
        ]
    
    return data_padded


def generate_short_term_pieces_from_dict(data):
    """This function generates he short term pieces for all tracks of the
    dataset
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'genre')

    Returns:
        data_pieces {dict} -- keys in ('train', 'test'), values are lists of 
            tuples ('file_name', 'numpy_representation', 'split_id', 'genre')
    """

    # track[0] is the "file_name"
    # piece[0] is the numpy representation of the chunk
    # piece[1] is the split id
    # track[2] is the "genre"

    params = load_params()

    # important to not overwrite `data`!
    data_pieces = {"train": None, "test": None}
    for split in data:
        data_pieces[split] = [
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
    
    return data_pieces


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
    frame_sample_length = int(
        params["mel_map"]["frame_length_in_s"] * params["sampling_rate"]
    )

    for split in mel_maps:
        mel_maps[split] = [
            (
                piece[0], 
                generate_mel_map(
                    piece[1], 
                    sampling_rate=params["sampling_rate"],
                    frame_length=frame_sample_length,
                    overlap=params["mel_map"]["overlap"],
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
    frame_sample_length = int(
        params["spectrogram"]["frame_length_in_s"] * params["sampling_rate"]
    )

    for split in spectrograms:
        spectrograms[split] = [
            (
                piece[0], 
                generate_spectrogram(
                    piece[1], 
                    frame_length=frame_sample_length, 
                    overlap=params["spectrogram"]["overlap"]
                ), 
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


def generate_glcms_from_dict(maps, map_type, serialize=False):
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
    
    glcms = []

    for angle in angles_in_deg:
        
        glcm_angle = {}
        
        for split in maps:

            # to have a serializable format, we just have to convert the gclm to 
            # a list format
            
            glcm_angle[split] = [
                (
                    piece[0], 
                    generate_glcm(
                        piece[1], 
                        distance=params["GLCM"]["distance"],
                        angle_in_deg=angle
                    ).tolist() if serialize
                    else 
                    generate_glcm(
                        piece[1], 
                        distance=params["GLCM"]["distance"],
                        angle_in_deg=angle
                    ), 
                    piece[2], 
                    piece[3]
                ) 
                for piece in maps[split]
            ]
            
        glcms.append(glcm_angle)
    
    return glcms


def drop_first_glcm_level_from_dict(glcms, serialize=False):
    """This function drops the first grey level from GLCMs and returns the dict
    with the smaller GLCMs 
    
    Arguments:
        glcms {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_glcm', 'split_id', 'genre')
        serialize {boolean} -- whether to return a list of lists from the numpy
            arrays or not

    Returns:
        output {list} -- list of dicts with keys in ('train', 'test'), and values
            lists of tuples ('file_name', 'glcm', 'split_id', 'genre'),
            one dict per angle
    """

    output = []

    for glcm in glcms:
        
        dropped_glcm = {split: None for split in glcm}

        for split in glcm:

            dropped_glcm[split] = [
                (
                    piece[0], 
                    piece[1][1:, 1:].tolist() if serialize 
                    else 
                    piece[1][1:, 1:],
                    piece[2], 
                    piece[3]
                ) 
                for piece in glcm[split]
            ]
        
        output.append(dropped_glcm)
    
    return output


def generate_MFCC_from_dict(data, serialize=False):
    """This function generates a spectrogram for all short term pieces of the
    dataset
    
    Arguments:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'numpy_representation', 'split_id', 'genre')

    Returns:
        data {dict} -- keys in ('train', 'test'), values are lists of tuples
            ('file_name', 'mfcc', 'split_id', 'genre')
        !!!! the output is the splitted mfcc so the shape is 
        (n_submaps, n_mfcc, t) which corresponds for the paper to (30, 40, 50)
    """

    params = load_params()

    mfcc = {"train": None, "test": None}

    for split in mfcc:
        mfcc[split] = [
            (
                piece[0],
                np.split(
                    generate_MFCC(
                        piece[1], 
                        n_mfcc =  params["MFCC"]["n_mfcc"],
                        frame_length=params["MFCC"]["frame_length_in_s"], 
                        overlap=params["MFCC"]["overlap"],
                        sampling_rate=params["sampling_rate"],
                        n_windows=params["MFCC"]["n_windows"]
                    ), 
                    params["MFCC"]["n_submaps"], 
                    axis=1
                ) if serialize else
                np.array(
                    np.split(
                        generate_MFCC(
                            piece[1], 
                            n_mfcc =  params["MFCC"]["n_mfcc"],
                            frame_length=params["MFCC"]["frame_length_in_s"], 
                            overlap=params["MFCC"]["overlap"],
                            sampling_rate=params["sampling_rate"],
                            n_windows=params["MFCC"]["n_windows"]
                        ),
                        params["MFCC"]["n_submaps"], 
                        axis=1
                    )
                ),
                piece[2]
            )
            for piece in data[split]
        ]
    
    return mfcc


def preprocess_data(pre_loaded_data=None, serialize=False):
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
    # --------------------------------------------------------------------------
    params = load_params()
    config = load_config()
    
    # STEP 2: Load the data
    # --------------------------------------------------------------------------

    if(pre_loaded_data is None):

        # read .wav files into np arrays
        data = read_in_data(
            params["sampling_rate"], sample_data=config["using_sample_data"]
        )

    else:
        data = pre_loaded_data

    # STEP 3: Pad the data to prepare the division into chunks
    # --------------------------------------------------------------------------

    data = pad_from_dict(data)

    # STEP 4: generate mfcc from the data
    # We need to do that before generating the short term pieces 
    # (because the process is different for mcff)
    # --------------------------------------------------------------------------

    mfcc = generate_MFCC_from_dict(data, serialize=serialize)

    # STEP 5: Cut the data into smaller chunks
    # --------------------------------------------------------------------------

    data = generate_short_term_pieces_from_dict(data)

    # STEP 6: Generate mel maps for each chunk
    # --------------------------------------------------------------------------

    # convert the numpy representations of .wav files to mel_maps
    mel_maps = generate_mel_maps_from_dict(data)

    # STEP 7: Generate spectrograms for each chunk
    # --------------------------------------------------------------------------

    # convert the numpy representations of .wav files to spectrograms
    spectrograms = generate_spectrograms_from_dict(data)

    # STEP 8: Quantize the maps
    # --------------------------------------------------------------------------

    quantized_mel_maps = generate_quantized_maps_from_dict(mel_maps)

    quantized_spectrograms = generate_quantized_maps_from_dict(spectrograms)

    # STEP 9: Generate the GLCMs
    # --------------------------------------------------------------------------

    # `serialize` = False in all cases, we serialize in the next step if need be
    glcms_from_spectrogram = generate_glcms_from_dict(
        quantized_spectrograms, map_type="spectrogram",
        serialize=False
    )


    glcms_from_mel_map = generate_glcms_from_dict(
        quantized_mel_maps, map_type="mel_map",
        serialize=False
    )

    # STEP 10: Drop the first level of GLCMs
    # --------------------------------------------------------------------------
    
    glcms_from_spectrogram = drop_first_glcm_level_from_dict(
        glcms_from_spectrogram, serialize=serialize
    )

    glcms_from_mel_map = drop_first_glcm_level_from_dict(
        glcms_from_mel_map, serialize=serialize
    )


    # return transformed data dictionary

    glcms = {
        "mfcc": mfcc,
        "spectrogram": glcms_from_spectrogram, 
        "mel_map": glcms_from_mel_map
    }

    return glcms


if __name__ == "__main__":

    config = load_config()

    print("Preprocessing the data")
    #Checking the preprocessing functions
    x = preprocess_data(serialize=True)            

    output_dir = git_root("data", "pipeline_output")

    train_test_list = ['train', 'test']

    print("Dumping the MFCC")

    full_or_sample = "full" if config["using_sample_data"] else "sample"

    mfcc_dir_path = git_root(
        "data", f"preprocessed_data_{full_or_sample}", "mfcc"
    )

    try:
        os.mkdir(mfcc_dir_path)
    except:
        pass

    for elem in train_test_list:
            filename = 'data_mfcc_{}.json'.format(elem)
            print(filename)

            #Dumping the result to a json file
            with open(os.path.join(mfcc_dir_path,filename),'w+') as outfile:
                json.dump(x['mfcc'][elem], outfile)
    
    #We dump the data
    print("Dumping the spectrogram data")

    spectrogram_dir_path = git_root(
        "data", f"preprocessed_data_{full_or_sample}", "spectrogram"
    )

    try:
        os.mkdir(spectrogram_dir_path)
    except:
        pass

    deg_list = config['feature_engineering']['GLCM']['spectrogram']["angles_in_deg"]
    for i in range(len(deg_list)):

        for elem in train_test_list:
            filename = 'data_spectrogram_angle_{}_{}.json'.format(deg_list[i], elem)
            print(filename)

            #Dumping the result to a json file
            with open(os.path.join(spectrogram_dir_path,filename),'w') as outfile:
                json.dump(x['spectrogram'][i][elem], outfile)


    print("Dumping the mel map data")

    mel_map_dir_path = git_root(
        "data", f"preprocessed_data_{full_or_sample}", "mel_map"
    )

    try:
        os.mkdir(mel_map_dir_path)
    except:
        pass

    deg_list = config['feature_engineering']['GLCM']['mel_map']["angles_in_deg"]
    for i in range(len(deg_list)):

        for elem in train_test_list:
            filename = 'data_mel_map_angle_{}_{}.json'.format(deg_list[i], elem)
            print(filename)

            #Dumping the result to a json file
            with open(os.path.join(mel_map_dir_path,filename),'w') as outfile:
                json.dump(x['mel_map'][i][elem], outfile)
    
