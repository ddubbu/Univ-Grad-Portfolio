import numpy as np
import os
import matplotlib.pyplot as plt
import librosa, librosa.display

from sklearn.model_selection import train_test_split


############ 2. Feature Extraction ############
# def plot_mfcc(mfcc, RECORD_FILE_NAME=None):
#     plt.figure()
#     librosa.display.specshow(mfcc, x_axis='time')  # y 축은 모르겠당 y_coords=13
#     # plt.plot(mfcc[1], mfcc[0])
#     plt.title('MFCC')
#     plt.xlabel('Time [s]')
#     plt.ylabel('MFCC Coefficeints')
#     plt.colorbar()
#     plt.show()

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

X_train = []  # train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []
num_class = 30
#num_mfcc = 24

def create_train_test_npy():
    inc_class = 0 # upto 29 (total 30개)
    #batch_waves = []
    #labels = []
    X_data = []
    Y_label = []
    # tf_classes = 0  #
    # global X_train, X_test, Y_train, Y_test, inc_class

    path = "C:/0. Git/speech_commands_v0.01"
    folders = os.listdir(path)
    # print(folders)
    # exit()
    for folder in folders:
        if not os.path.isdir(path): continue  # 폴더가 아니면 continue
        #if folder[0:2] == "_b": pass  # background pass
        files = os.listdir(path + "/" + folder)  # stop 같은 파일
        print("Foldername :", folder, "-", len(files), "파일")
        # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력

        temp = 1
        how_many = 1000
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            else:
                if temp == how_many + 1:
                    print("%d개 끝"%(how_many))
                    print(np.shape(X_data), np.shape(Y_label))
                    break
                #print("num%d"%temp)
                temp += 1
                # print("Filename :",wav)#.wav 파일이 아니면 continue
                # y, sr = librosa.load(path + "/" + folder + "/" + wav)
                wav_path = path + "/" + folder + "/" + wav
                mfcc = feature_mfcc(wav_path)  # (24, 173)

                if(mfcc.shape != (24, 173)):
                    print("MFCC 에러")
                    exit()
                X_data.append(mfcc.T)  # (173, 24)
                # print('mfcc.shape = ', mfcc.shape)
                # print(len(mfcc))

                #label = np.zeros(shape=(1, num_class))


                label = [0 for i in range(num_class)]  # ★ call by reference 걱정 안해도 됨. 리스트 새로 생성됨
                label[inc_class] = 1
                Y_label.append(label)

                # for i in range(len(mfcc.T)):
                #     Y_label.append(label)
                #print(label)
                # print(np.shape(X_data), np.shape(Y_label))
                # exit()
        inc_class = inc_class + 1
    # end loop
    print("X_data :", np.shape(X_data))
    print("Y_label :", np.shape(Y_label))
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_data), np.array(Y_label), test_size=0.1, shuffle=True)


    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./data.npy", xy)

    print("make numpy file ========== clear =============")
    print("folders for class : %d 개"%len(folders))
    print(folders)


################## MAKE DATA ##############

# make train data
create_train_test_npy()


# folders =
# ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
