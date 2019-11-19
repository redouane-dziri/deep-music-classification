import numpy as np

import librosa
import sys
import os
from git_root import git_root

# <---- For importing a .py file from another module ---->
sys.path.append(os.path.join(git_root(), "utils"))
from utils import read_in_data, generate_short_term_piece
from utils import quantize, load_params, load_config



def generate_MFCC(array, n_mfcc, frame_length, overlap, sampling_rate, n_windows):
    """This function generates a MFCC 
    from a numpy representation of mono .wav files

    <---- WARNING: the number of windows computed is a FIXED parameter from the 
    config file ---->
    
    Arguments:
        array {np.array} -- float np.array
        frame_length {int} -- the number of samples in each analysis window
        overlap {float} -- in [0, 1) the fraction of overlap for each window

    returns: np.array with dimensions (n_mfcc, t)
    """

    window_length= int(frame_length * sampling_rate)
    hop_length = int(window_length * (1 - overlap))

    # we have to pad the array before computing the mfcc
    length_needed = int(n_windows * hop_length) + window_length


    # number of values we need to add to the vector
    padding = hop_length * n_windows + window_length - 1 - array.shape[0]

    if(array.shape[0] < length_needed):
        # we need to pad the array to account for the last incomplete window
        padded_array = np.pad(
            array, (0,padding), mode='constant', constant_values=(0,0)
        )
    else:
        padded_array = array[0:length_needed-1]

    mfcc = librosa.feature.mfcc(padded_array, 
                                sr=sampling_rate, 
                                n_mfcc=n_mfcc,
                                center=False,
                                n_fft=window_length, 
                                hop_length=hop_length)

    return mfcc

def generate_formatted_MFCC(array, n_mfcc, frame_length, overlap, sampling_rate, n_windows):
    """This function generates a MFCC from a numpy representation of mono .wav files
    in the correct format for the neural net.
    In addition to generating the mfcc, the mfcc is also splitted across axis=1,
    and turned into a 3D array.

    <---- WARNING: the number of windows computed is a FIXED parameter from the 
    config file ---->
    
    Arguments:
        array {np.array} -- float np.array
        frame_length {int} -- the number of samples in each analysis window
        overlap {float} -- in [0, 1) the fraction of overlap for each window

    returns: np.array with dimensions (nb_mfcc,t,n_windows)
    """

    mfcc = generate_MFCC(array, n_mfcc, frame_length, overlap, sampling_rate, n_windows)
    mfcc_formatted = np.array(np.split(mfcc,30,axis=1))
    return mfcc_formatted

if __name__=="__main__":
    params = load_params()
    config = load_config()
    data = read_in_data(
            params["sampling_rate"], sample_data=config["using_sample_data"]
        )
    sample_array = data['test'][5][1]
    sample_array = np.array(sample_array.tolist() + list(range(10000)))

    x = generate_MFCC(sample_array,
                        n_mfcc =  params["MFCC"]["n_mfcc"],
                        frame_length=params["MFCC"]["frame_length_in_s"], 
                        overlap=params["MFCC"]["overlap"],
                        sampling_rate=params["sampling_rate"],
                        n_windows=params["MFCC"]["n_windows"])

    #Output format should be (40, 1500)
    print(x.shape)

    y = generate_formatted_MFCC(sample_array,
                        n_mfcc =  params["MFCC"]["n_mfcc"],
                        frame_length=params["MFCC"]["frame_length_in_s"], 
                        overlap=params["MFCC"]["overlap"],
                        sampling_rate=params["sampling_rate"],
                        n_windows=params["MFCC"]["n_windows"])

    #Output format should be (30, 40, 50)
    print(y.shape)

