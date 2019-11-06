import numpy as np

from librosa.feature import melspectrogram


def generate_mel_map(array, sampling_rate, hop_length, n_mels):
    """This function generates mel maps from a numpy representation of mono .wav
    files
    
    Arguments:
        array {np.array} -- float np.array
        hop_length {int} -- the hop length in units of frames per second
        n_mels {int} -- the number of filter-bank channels
    """

    mel_map = melspectrogram(
        array, 
        sr=sampling_rate,
        hop_length=hop_length, 
        center=False, 
        n_mels=n_mels
    )

    return mel_map
