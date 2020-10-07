from model.Classifier.ResNet import *
from model.KWS.GRU_2layer import *
from wav_utils import *
import os
from torch.autograd import Variable

'''torch.device'''
device = torch.device('cpu')

def KWS(RECORD_FILE_PATH, KWS_model):

    n_mfcc = 24
    Tx = 1723
    num_class = 30

    ''' 모델 불러오기 '''
    KWS_model.eval()

    mfcc = feature_mfcc(RECORD_FILE_PATH, mode="KWS")  # [n_mfcc, Tx]
    mfcc = np.transpose(mfcc)  # [Tx, n_mfcc]
    mfcc = np.expand_dims(mfcc, 0)  # [1, Tx, n_mfcc]
    mfcc = torch.Tensor(mfcc)
    prediction = KWS_model(mfcc).squeeze()

    result = prediction.detach().cpu().numpy()
    sig_result = torch.sigmoid(prediction).detach().cpu().numpy()

    fig_probability = plt.figure(figsize=(4, 7))
    ax1 = fig_probability.add_subplot(2, 1, 1)
    plt.plot(result, 'b')
    ax1.set(title='Before Sigmoid', ylabel="Predict", xlabel='Tx_segment')

    ax2 = fig_probability.add_subplot(2, 1, 2)
    plt.plot(sig_result, 'b')
    ax2.set(title='After Sigmoid', ylabel="Probability", xlabel='Tx_segment')
    plt.tight_layout()  # subplot끼리 안 곂치게
    plt.savefig("./dev_output/KWS_Probability.jpg")
    peak_y = np.max(result)
    peak_x = np.where(result == peak_y)[0][0]

    # get Extraction duration
    T_y = 1723
    mark_length = 300  # 172로 하면, Tight 하긴한데 피아노 배경 Activate 말고 학습 굿
    left_step = int(mark_length * 0.3)  # 0.5의 휴유증이 세다..
    left_x = peak_x - left_step if peak_x - left_step > 0 else 0
    right_x = left_x + mark_length

    # pyaudio peak and restore
    # pydub does thing in milliseconds
    segment_left_t = int(left_x * 10000.0 / T_y)  # + 2500
    segment_right_t = int(right_x * 10000.0 / T_y)  # + 2500

    print("time: ", segment_left_t, "~", segment_right_t)

    sig = AudioSegment.from_wav(RECORD_FILE_PATH)

    ex_Keyword = sig[segment_left_t: segment_right_t + 1]

    ex_FILE_PATH = "./keyword.wav"
    file_handle = ex_Keyword.export(ex_FILE_PATH, format="wav")

    '''## Extraction plotting'''

    scatter_x = [peak_x]
    scatter_y = [peak_y]

    fig_extraction = plt.figure(figsize=(4, 7))
    ax1 = fig_extraction.add_subplot(2, 1, 1)
    plt.plot(result)
    plt.scatter(scatter_x, scatter_y, c=['r'])
    plt.axvline(x=left_x, color='r', linestyle='-')
    plt.axvline(x=right_x, color='r', linestyle='-')
    ax1.set(title='[Predict] Keyword Position', ylabel="Predict", xlabel='Tx_segment')

    sig, sr = librosa.load(RECORD_FILE_PATH)
    Time = np.linspace(0, len(sig) / sr, num=len(sig))

    ax2 = fig_extraction.add_subplot(2, 1, 2)
    plt.plot(Time, sig)
    plt.axvline(x=segment_left_t / 1000, color='r', linestyle='-')
    plt.axvline(x=segment_right_t / 1000, color='r', linestyle='-')
    ax2.set(title='[Wav Form] Keyword Position', ylabel="Amplitude", xlabel='sec')
    plt.tight_layout()  # subplot끼리 안 곂치게
    plt.savefig("./dev_output/KWS_Keyword_Position.jpg")
    #plt.show()
    print("here2")

    return ex_FILE_PATH

classes = ['bed', 'bird', 'cat', 'dog', 'down',
           'eight', 'five', 'four', 'go', 'happy',
           'house', 'left', 'marvin', 'nine', 'no',
           'off', 'on', 'one', 'right', 'seven',
           'sheila', 'six', 'stop', 'three', 'tree',
           'two', 'up', 'wow', 'yes', 'zero']

def activation(ex_FILE_PATH):
    n_mfcc = 24
    Tx = 173
    num_class = 30

    ''' Extract 1s words in 10s '''
    RATE = 44100

    mfcc = feature_mfcc(ex_FILE_PATH, mode="Classifier")
    mfcc = mfcc.transpose()
    # 최종 hope shape -> (batch_szie=1, channel=1, 173, 24)
    mfcc = np.expand_dims(mfcc, axis=(0))
    x = np.expand_dims(mfcc, axis=(0))
    print(x.shape)

    x = Variable(torch.Tensor(x))

    ''' Load Model '''
    model_save_type = ["all", "state_dict", "ckp"]  # 학습 재개를 위해서 ckp 저장 권장 // else state_dict
    model_name = "./model/Classifier/Classifier_ResNet"

    # model 초기화
    model = ResNet(ResidualBlock, [2, 2, 2])
    model_path = model_name + '_ckp.tar'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    ''' Audio Test '''

    model.eval()
    # model.cuda()

    # input shape check
    if x.shape == (1, 1, 173, 24):
        print("activation shape 통과")

    else:
        print("activation shape 이상")

    with torch.no_grad():
        output = model(x)  # output.shape = (1, 30)
        prediction = output.argmax(dim=1, keepdims=True)
        print("prediction :", end=' ')
        for i in range(len(prediction)):
            print(classes[prediction[i]], end=' ')
        print()

    output = output.squeeze().detach().cpu().numpy()
    threshold = np.exp(output[22])

    # print("========= other probability ==========")
    # print(np.exp(output))

    return threshold, np.exp(output)

##########################################################################
def STT(RECORD_FILE_PATH):
    n_mfcc = 24
    Tx = 173
    num_class = 30

    ''' Extract 1s words in 10s '''
    RATE = 44100

    spf = wave.open(RECORD_FILE_PATH, 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, dtype=np.int16)
    # 시간 흐름에 따른 그래프를 그리기 위한 부분
    Time = np.linspace(0, len(signal) / RATE, num=len(signal))  # len(signal)/RATE
    # print(len(Time))
    fig1 = plt.figure()
    plt.title('Voice Signal Wave...')
    peaks_idxs, _ = find_peaks(signal, height=(2000, 7000),
                               distance=50000)  # distance 늘리면, peak 1개씩만 갖다주구만!  # min, max 정해줘야하다니...
    # 10s_one_stop_two.wav 기준으로 db 5000 이상! >> 음원 자체를 크게 말하시오.

    print("len(peaks_idxs): ", len(peaks_idxs))
    print("peaks_idx: ", peaks_idxs)

    peak_signal = signal.copy()
    for idx in range(len(signal)):
        if idx in peaks_idxs:
            continue
        else:
            peak_signal[idx] = 0

    plt.plot(Time, signal, Time, peak_signal)  # 같이 plot하는 아이들의 x 길이 맞춰주자!

    plt.savefig("./dev_output/STT_Peak_Detection.jpg")
    #plt.show()

    # dev_output/STT_extractions 파일 포맷
    remove_files = os.listdir('dev_output/STT_extractions/')
    for file in remove_files:
        os.remove('./dev_output/STT_extractions/' + file)

    print("FILE REMOVE===> ./dev_output/STT_extractions/")
    os.makedirs('dev_output/STT_extractions/', exist_ok=True)

    # extract_1s = []
    pre_ex_file = 'dev_output/STT_extractions/'  # 파일을 아예 만들자 STT_extractions/
    plt.plot(Time, signal)
    mark_length = RATE  # 1초만 자를거임.
    cnt = 0
    for idx in peaks_idxs:
        left_x = idx - mark_length / 2 if idx - mark_length / 2 > 0 else 0
        right_x = idx + mark_length / 2 - 1

        segment_left_t = left_x / RATE  # int(left_x / RATE)
        segment_right_t = right_x / RATE  # int(right_x / RATE)
        plt.axvline(x=segment_left_t, color='r', linestyle='-')
        plt.axvline(x=segment_right_t, color='r', linestyle='-')

        raw_sig = AudioSegment.from_wav(RECORD_FILE_PATH)

        print("segment perid: \n", segment_left_t * 1000, '~', segment_right_t * 1000)
        ex_word = raw_sig[segment_left_t * 1000:segment_right_t * 1000 + 1]
        ex_file_path = pre_ex_file + str(cnt) + '.wav'

        file_handle = ex_word.export(ex_file_path, format='wav')


        cnt = cnt + 1
    plt.savefig("./dev_output/STT_extraction_position.jpg")
    #plt.show()

    print("1초씩 분리 완료")

    '''make input mfcc like batch shape'''
    ex_files = os.listdir(pre_ex_file)
    ex_files.sort()  # 오름차순 주의
    print(ex_files)

    ex_mfccs = []
    for ex_file in ex_files:
        audio_path = pre_ex_file + ex_file

        # mfcc.shape = (24, 173)
        print(audio_path)
        mfcc = feature_mfcc(audio_path, mode="Classifier")
        mfcc = mfcc.transpose()
        ex_mfccs.append(mfcc)

    # 최종 hope shape -> (batch_szie=1, channel=1, 173, 24)
    x = np.expand_dims(ex_mfccs, axis=(1))
    print(x.shape)

    x = Variable(torch.Tensor(x))

    ''' Load Model '''
    model_save_type = ["all", "state_dict", "ckp"]  # 학습 재개를 위해서 ckp 저장 권장 // else state_dict
    model_name = "./model/Classifier/Classifier_ResNet"
    option = 2

    if option == 0:
        # 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
        model_path = model_name + '_all.pt'
        model = torch.load(model_path)

    elif option == 1:
        model_path = model_name + '_state_dict.pt'
        model = ResNet()
        model.load_state_dict(torch.load(model_path))

    elif option == 2:
        # model, optimizer 초기화
        model = ResNet(ResidualBlock, [2, 2, 2])
        # optimizer = optim.Adam(model.parameters(), lr=lr)

        model_path = model_name + '_ckp.tar'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']
        loss = checkpoint['loss']

    ''' Audio Test '''



    model.eval()

    sentence = []
    with torch.no_grad():
        output = model(x)  # output.shape = (1, 30)
        prediction = output.argmax(dim=1, keepdims=True)
        print("prediction :", end=' ')
        for i in range(len(prediction)):
            print(classes[prediction[i]], end=' ')
            sentence.append(classes[prediction[i]])
        print()

    return sentence