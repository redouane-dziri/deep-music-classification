import numpy as np

from scipy.signal import spectrogram


def generate_spectrogram(array, hop_length):
    """This function generates a spectrogram with consecutive Fourier transforms
    from a numpy representation of mono .wav files
    
    Arguments:
        array {np.array} -- float np.array
        hop_length {int} -- the hop length in units of frames per second
        n_mels {int} -- the number of filter-bank channels
    """

    _, _, spect = spectrogram(
        array,
        nperseg=hop_length,
        noverlap=hop_length//2
    )

    return spect
