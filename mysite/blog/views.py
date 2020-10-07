############################## Start ##############################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision.transforms import transforms  # 1 batch = (1, 784)
# from torch.autograd import Variable
# from torch.utils.data.dataloader import DataLoader
# from matplotlib import pyplot as plt
# import numpy as np
#
# from modeling.ResNet import *
# from wav_utils import *
# import os

############################## End ##############################

from django.shortcuts import render
from .models import Category, Post
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import CreateView

# Create your views here.
def index(request):
    post_latest = Post.objects.order_by("-createDate")[:6]
    context = {
        "post_latest": post_latest
    }

    return render(request, 'blog/index.html', context=context)


class PostDetailView(generic.DetailView):
    model = Post


class PostCreate(LoginRequiredMixin, CreateView):
    model = Post
    fields = ["title", "title_image", "title_music", "content", "category"]

############################## Start ##############################

# def output(request):
#     # ''' 10s 녹음 후 저장 하는 소스코드'''
#     # ##
#     #
#     # '''torch.device'''
#     # print("is there cuda? :", torch.cuda.is_available())
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # print(device)
#     #
#     # n_mfcc = 24
#     # Tx = 173
#     # num_class = 30
#     #
#     # ''' Extract 1s words in 10s '''
#     # audio_path = './dev_word/10s_one_stop_two.wav'  # 10s_one_stop_two.wav  # 10s_stop_one.wav
#     # # ipd.Audio(audio_path)
#     #
#     # RATE = 44100
#     #
#     # spf = wave.open(audio_path, 'r')
#     # signal = spf.readframes(-1)
#     # signal = np.fromstring(signal, dtype=np.int16)
#     # # 시간 흐름에 따른 그래프를 그리기 위한 부분
#     # Time = np.linspace(0, len(signal) / RATE, num=len(signal))  # len(signal)/RATE
#     # # print(len(Time))
#     # fig1 = plt.figure()
#     # plt.title('Voice Signal Wave...')
#     # peaks_idxs, _ = find_peaks(signal, height=(6000, 7000),
#     #                            distance=50000)  # distance 늘리면, peak 1개씩만 갖다주구만!  # min, max 정해줘야하다니...
#     # # 10s_one_stop_two.wav 기준으로 db 5000 이상! >> 음원 자체를 크게 말하시오.
#     #
#     # print("len(peaks_idxs): ", len(peaks_idxs))
#     # print("peaks_idx: ", peaks_idxs)
#     #
#     # peak_signal = signal.copy()
#     # for idx in range(len(signal)):
#     #     if idx in peaks_idxs:
#     #         continue
#     #     else:
#     #         peak_signal[idx] = 0
#     #
#     # plt.plot(Time, signal, Time, peak_signal)  # 같이 plot하는 아이들의 x 길이 맞춰주자!
#     #
#     # plt.savefig("./dev_word/Voice_Signal_Wave.jpg")
#     # plt.show()
#     #
#     # # extract_1s = []
#     # pre_ex_file = './dev_word/extraction/'  # 파일을 아예 만들자 extraction/
#     # plt.plot(Time, signal)
#     # mark_length = RATE  # 1초만 자를거임.
#     # cnt = 0
#     # for idx in peaks_idxs:
#     #     left_x = idx - mark_length / 2 if idx - mark_length / 2 > 0 else 0
#     #     right_x = idx + mark_length / 2 - 1
#     #
#     #     segment_left_t = left_x / RATE  # int(left_x / RATE)
#     #     segment_right_t = right_x / RATE  # int(right_x / RATE)
#     #     plt.axvline(x=segment_left_t, color='r', linestyle='-')
#     #     plt.axvline(x=segment_right_t, color='r', linestyle='-')
#     #
#     #     raw_sig = AudioSegment.from_wav(audio_path)
#     #
#     #     print("segment perid: \n", segment_left_t * 1000, '~', segment_right_t * 1000)
#     #     ex_word = raw_sig[segment_left_t * 1000:segment_right_t * 1000 + 1]
#     #     ex_file_path = pre_ex_file + str(cnt) + '.wav'
#     #
#     #     file_handle = ex_word.export(ex_file_path, format='wav')
#     #
#     #     # ipd.Audio(ex_file_path)
#     #
#     #     cnt = cnt + 1
#     #
#     # plt.savefig("./dev_word/Segment.jpg")
#     # plt.show()
#     #
#     # print("1초씩 분리 완료")
#     #
#     # '''make input mfcc like batch shape'''
#     # ex_files = os.listdir(pre_ex_file)
#     # ex_files.sort()  # 오름차순 주의
#     # print(ex_files)
#     #
#     # ex_mfccs = []
#     # for ex_file in ex_files:
#     #     audio_path = pre_ex_file + ex_file
#     #
#     #     # mfcc.shape = (24, 173)
#     #     print(audio_path)
#     #     mfcc = feature_mfcc(audio_path)
#     #     mfcc = mfcc.transpose()
#     #     ex_mfccs.append(mfcc)
#     #
#     # # 최종 hope shape -> (batch_szie=1, channel=1, 173, 24)
#     # x = np.expand_dims(ex_mfccs, axis=(1))
#     # print(x.shape)
#     #
#     # x = Variable(torch.Tensor(x)).to(device)
#     #
#     # ''' Load Model '''
#     # model_save_type = ["all", "state_dict", "ckp"]  # 학습 재개를 위해서 ckp 저장 권장 // else state_dict
#     # model_name = "./modeling/ResNet"
#     # option = 2
#     #
#     # if option == 0:
#     #     # 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
#     #     model_path = model_name + '_all.pt'
#     #     model = torch.load(model_path)
#     #
#     # elif option == 1:
#     #     model_path = model_name + '_state_dict.pt'
#     #     model = ResNet().to(device)
#     #     model.load_state_dict(torch.load(model_path))
#     #
#     # elif option == 2:
#     #     # model, optimizer 초기화
#     #     model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#     #     # optimizer = optim.Adam(model.parameters(), lr=lr)
#     #
#     #     model_path = model_name + '_ckp.tar'
#     #     checkpoint = torch.load(model_path, map_location='cpu')
#     #     model.load_state_dict(checkpoint['model_state_dict'])
#     #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     #     last_epoch = checkpoint['last_epoch']
#     #     loss = checkpoint['loss']
#     #
#     # ''' Audio Test '''
#     #
#     # classes = ['bed', 'bird', 'cat', 'dog', 'down',
#     #            'eight', 'five', 'four', 'go', 'happy',
#     #            'house', 'left', 'marvin', 'nine', 'no',
#     #            'off', 'on', 'one', 'right', 'seven',
#     #            'sheila', 'six', 'stop', 'three', 'tree',
#     #            'two', 'up', 'wow', 'yes', 'zero']
#     #
#     # model.eval()
#     # # model.cuda()
#     #
#     # with torch.no_grad():
#     #     output = model(x)  # output.shape = (1, 30)
#     #     prediction = output.argmax(dim=1, keepdims=True)
#     #     print("prediction :", end=' ')
#     #     for i in range(len(prediction)):
#     #         print(classes[prediction[i]], end=' ')
#     result = Post.title
#     return render(request, 'post_record.html', {'data:' result})
#
# def button(request):
#     return render(request, 'post_record.html')
############################## End ##############################
