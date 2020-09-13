import librosa
import wave
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks



def match_1s(data):
    # 30sec 길이 통일
    num = 22050
    # print(data.shape, end=' ->')
    if len(data) < 22050:
        num = 22050 - len(data)
        temp = np.zeros(num) #* 1e-05
        data = np.append(data, temp)
    elif len(data) > 22050:
        data = data[:22050]

    #print(data.shape, end=' ')

    # (22050,) to column vector : (2250, 1)
    # data = data.reshape(len(data), 1)

    return data


def feature_mfcc(RECORD_FILE_NAME):

    # 조정할 수 있는 건 다 적어보자.

    # sr = 22050 = bitrate/2 -> Q. bitrate 와 어떤 관계?
    # Generate mfccs from a time series
    # t초당 sig.shape = (t*sr,)
    sig, sr = librosa.load(RECORD_FILE_NAME)  # , sr=sr
    # 만약, sr=16000, mfcc.shape = (n_mfcc,1251)
    #       sr=(default)22050, mfcc.shape = (n_mfcc, 1723)


    hop_length = 0
    if len(sig) == 22050:  # 128 -> mfcc Tx 301, 223 -> mfcc Tx 173
        hop_length = 128
    elif len(sig) == 38433:
        hop_length = 223  # Tx 173 으로 통일
    elif len(sig) < 22050:
        # print("smaller than", end=' ')
        sig = match_1s(sig)
        hop_length = 128

    else:
        # print(len(sig))
        sig = match_1s(sig)
        print("1s over")
        hop_length = 128

    n_mfcc = 24
    # n_mels = 20
    n_fft = 101
    fmin = 0
    fmax = None
    # sr = 16000


    mfcc = librosa.feature.mfcc(y=sig, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)
    # print("here", mfcc.shape)

    return mfcc