import librosa
import numpy as np

x = np.arange(0,4,0.5)
print(x)
window_length = 2
hop_length = 1

x_windowed = librosa.feature.mfcc(x,
                                n_mfcc=1,
                                center=True,
                                n_fft=window_length, 
                                hop_length=hop_length)
print(x_windowed.shape)

x_windowed = librosa.feature.mfcc(x,
                                n_mfcc=1,
                                center=False,
                                n_fft=window_length, 
                                hop_length=hop_length)
print(x_windowed.shape)