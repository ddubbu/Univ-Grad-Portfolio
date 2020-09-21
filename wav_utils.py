import wave
import matplotlib.pyplot as plt
import numpy as np
import librosa, librosa.display
from scipy.signal import find_peaks # inference에서 사용됨.
from pydub import AudioSegment  # inference에서 사용됨.
import pyaudio  # 마이크 사용
from datetime import datetime  # 저장될 파일 이름


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정 [bps]
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 10  # 녹음할 시간 설정
# WAVE_OUTPUT_FILENAME = "record.wav"

############ 1. 오디오 녹음 ############
# record 10s and store wav file
# Q. chunk, br과 sr 관계는 나중에 이해하자.
def record(FILENAME):
    print("record start!")
    # 파일명 조심 : 파일명에 콜론 들어가면 안됨
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-')
    WAVE_OUTPUT_FILENAME = "record_data/" + now + str(FILENAME) + ".wav"
    print(WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()  # 오디오 객체 생성

    stream = p.open(format=FORMAT,  # 16비트 포맷
                    channels=CHANNELS, #  모노로 마이크 열기
                    rate=RATE, #비트레이트
                    input=True,
                    # input_device_index=1,
                    frames_per_buffer=CHUNK)
                      # CHUNK만큼 버퍼가 쌓인다.

    print("Start to record the audio.")

    frames = []  # 음성 데이터를 채우는 공간
    print_sec = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

        if i == RATE/CHUNK * print_sec :
            print("Recording.... ", print_sec + 1, "s")
            print_sec += 1

    print("Recording is finished.")

    stream.stop_stream() # 스트림닫기
    stream.close() # 스트림 종료
    p.terminate() # 오디오객체 종료

    # WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    # 음성 wave plot
    spf = wave.open(WAVE_OUTPUT_FILENAME,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, dtype=np.int16)

    # 시간 흐름에 따른 그래프를 그리기 위한 부분
    Time = np.linspace(0,len(signal)/RATE, num=len(signal))

    fig1 = plt.figure()
    plt.title('Voice Signal Wave...')
    plt.plot(Time, signal)
    plt.show()
    plt.close(fig1)  # 닫아줘야하는 번거로움
    print("record end!!")

    return WAVE_OUTPUT_FILENAME


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



def feature_mfcc(RECORD_FILE_NAME, mode=None):

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
        if mode == "Classifier":
            sig = match_1s(sig)
            print("1s over")

        # else pass
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