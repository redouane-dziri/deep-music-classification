import numpy as np

import librosa 

from generate_spectrogram import generate_spectrogram


def generate_mel_map(array, sampling_rate, frame_length, overlap, n_mels):
    """This function generates mel maps from a numpy representation of mono .wav
    files
    
    Arguments:
        array {np.array} -- float np.array
        sampling_rate {int} -- the sampling rate of the array computed from the
            audio file
        overlap {float} -- in [0, 1) the fraction of overlap for each window
        n_mels {int} -- the number of filter-bank channels (ie. the number of
            bins of the frequency scale)
    """

    mel_map = librosa.feature.melspectrogram(
        array, 
        sr=sampling_rate, 
        n_fft=frame_length, 
        hop_length=int((1 - overlap) * frame_length), 
        n_mels=n_mels
    )
    mel_map_dB = librosa.power_to_db(mel_map, ref=np.max)

    return mel_map_dB
