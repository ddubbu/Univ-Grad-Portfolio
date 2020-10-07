import numpy as np
import os
from pydub import AudioSegment  # for manipulating audio data.
import matplotlib.pyplot as plt

import librosa, librosa.display

import pyaudio  # 마이크 사용
import wave  # 녹음한 데이터 wav 파일로 저장.
from datetime import datetime  # 저장될 파일 이름

Ty = 1723


def feature_mfcc(RECORD_FILE_NAME):

    # 조정할 수 있는 건 다 적어보자.
    hop_length= 128
    n_mfcc = 24
    # n_mels = 20
    n_fft = 101
    fmin = 0
    fmax = None
    # sr = 16000

    # sr = 22050 = bitrate/2 -> Q. bitrate 와 어떤 관계?
    # Generate mfccs from a time series
    # t초당 sig.shape = (t*sr,)
    sig, sr = librosa.load(RECORD_FILE_NAME)  # , sr=sr

    # 만약, sr=16000, mfcc.shape = (n_mfcc,1251)
    #       sr=(default)22050, mfcc.shape = (n_mfcc, 1723)


    mfcc = librosa.feature.mfcc(y=sig, sr=sr, hop_length=hop_length, fmin=fmin, fmax = fmax,
                                  n_fft= n_fft, n_mfcc=n_mfcc)
    print("here", mfcc.shape)

    return mfcc

# Load raw audio files for speech synthesis
def load_raw_audio():
    PATH = "C:/0. Git/speech_commands_v0.01"
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(PATH + "/_stop"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(PATH + "/_stop/"+filename)
            activates.append(activate)
    for filename in os.listdir(PATH + "/_backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(PATH + "/_backgrounds/"+filename)
            backgrounds.append(background)

    ## negative
    listdir = os.listdir(PATH)[:-3]  # _ 붙은 앞에 3개 버리기
    number_of_negatives = 100  # 100개 다른 단어 추출
    random_indices = np.random.randint(len(listdir), size=number_of_negatives)  # 중복될 듯
    for i in random_indices:
        select_folder = listdir[i]
        select_PATH = PATH + "/" + select_folder
        wave_list = os.listdir(select_PATH)
        #print(select_PATH)
        select_wave_idx = np.random.randint(len(wave_list), size=1)
        #print(select_wave_idx)
        select_wave_PATH = select_PATH + "/" + wave_list[select_wave_idx[0]]

        negative = AudioSegment.from_wav(select_wave_PATH)
        negatives.append(negative)
    return activates, negatives, backgrounds


def is_overlapping(new_segment, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    previous_segments could be many.
    """
    segment_start, segment_end = new_segment
    # Step 1: Initialize overlap as a "False" flag.
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    for previous_start, previous_end in previous_segments:  # it has a tuple data.
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap  # Bool

def get_random_time_segment(segment_ms):
    """
    find random position for len(segment_ms) in a 10,000 ms(=10s) audio clip.
    segment_ms -- the duration of the audio clip in ms
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

def insert_audio_clip(background, new_audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step,
    ensuring that new audio segment does not overlap with existing segments.
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(new_audio_clip)

    # Step 1: Pick a random time segment onto which to insert new audio clip.
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if overlap with previous segments.
    # ★★★  Warning : Infinite Loop ★★★
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add previous list not to overlap
    previous_segments.append(segment_time)

    # Step 4: Insert audio segments and background
    new_background = background.overlay(new_audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(Ty, y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_start_y should be 0 while, the
    50 following labels should be ones.

    ★ ★ Q. Why?  50 labels??
        len(new_segment) 만큼 아님?
        그리고 왜 끝나는 지점부터 50 segment를 하냐고....


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)  # = 100

    # Add 1 to the correct index in the background label (y)
    mark_length = 172
    for i in range(segment_end_y + 1, segment_end_y + 100 + 1):
        if i < Ty:
            y[0, i] = 1

    return y

def create_training_examples(Ty, num_new_data, folder_path):
    """
    It creates the whole training data
    and saves it in X.npy and Y.npy.
    """
    # Load audio segments
    activates, negatives, backgrounds = load_raw_audio()
    X = []
    Y = []
    for i in range(num_new_data):
        x, y = create_training_example(Ty, backgrounds, activates, negatives, "train"+str(i))
        X.append(x.T)
        Y.append(y.T)
    X = np.array(X)
    Y = np.array(Y)
    np.save(file=folder_path + '_X.npy', arr=X)
    np.save(file=folder_path + '_Y.npy', arr=Y)
    print("The end of making # of", num_new_data, "new training audios")


def create_training_example(Ty, backgrounds, activates, negatives, file_name_added):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    print("file num:", file_name_added)
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Step 3: # Select 0-1 random "background" audio clips from the entire list of "backgrounds" recordings
    random_indices = np.random.randint(len(backgrounds))
    background = backgrounds[random_indices]
    # Make background quieter
    background = background - 20  # ★ 왜?

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = 1  # ★★★ 한개만 넣자  //np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 4: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        print("activate insert time :", segment_time)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(Ty, y, segment_start)  # 첫 지점으로
        # y = insert_ones(Ty, y, segment_end)


    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = 2  # ★★★ 얘도 한개만 넣자. //np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]


    # # Step 5: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, n_segment_time = insert_audio_clip(background, random_negative, previous_segments)
        print("negative insert time :", n_segment_time)
    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)  # ★

    # Export new training example
    file_name = "./generate_data/" + str(file_name_added) + ".wav"
    file_handle = background.export(file_name, format="wav")
    print("File was saved in ./generate_data directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = feature_mfcc(file_name)

    # print("=============\nx", x.shape)
    # print("=============\ny", y.shape, "\n", y)
    # plot_mfcc(x)
    # plt.figure()
    # plt.plot(y[0])
    # plt.show()

    return x, y

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

######################## MAKE DATA ########################
# make train data
for i in range(50):  # *.np file
    np_folder_path = './XY_train/train'+str(i)
    create_training_examples(Ty, 100, np_folder_path)
    print("======= %d 개 완료======="%(i*100))