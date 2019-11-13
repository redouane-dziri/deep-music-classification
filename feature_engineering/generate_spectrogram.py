import numpy as np

import librosa


def generate_spectrogram(array, frame_length, overlap):
    """This function generates a spectrogram with consecutive Fourier transforms
    from a numpy representation of mono .wav files
    
    Arguments:
        array {np.array} -- float np.array
        frame_length {int} -- the number of samples in each analysis window
        overlap {float} -- in [0, 1) the fraction of overlap for each window
    """

    fourier_transform = np.abs(
        librosa.stft(
            array, n_fft=frame_length, hop_length=int((1 - overlap) * frame_length)
        )
    )

    # convert the amplitude spectrogram to dB-scaled spectrogram
    # scaling with respect to the maximum value of the Fourier transform
    spectrogram = librosa.amplitude_to_db(fourier_transform, ref=np.max)

    return spectrogram
