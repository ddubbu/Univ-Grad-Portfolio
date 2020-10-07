## Test code

''' import libraries '''
from inference import *

classes = ['bed', 'bird', 'cat', 'dog', 'down',
           'eight', 'five', 'four', 'go', 'happy',
           'house', 'left', 'marvin', 'nine', 'no',
           'off', 'on', 'one', 'right', 'seven',
           'sheila', 'six', 'stop', 'three', 'tree',
           'two', 'up', 'wow', 'yes', 'zero']

# KWS 모델을 먼저 불러볼까?
print("KWS model load 중 ......")
KWS_model_name = "./model/KWS/KWS_GRU_2layer"

# model, optimizer 초기화
KWS_model = KWS_2GRU()  # .to(device)
KWS_model_path = KWS_model_name + '_ckp.tar'
checkpoint = torch.load(KWS_model_path, map_location=device)
KWS_model.load_state_dict(checkpoint['model_state_dict'])
print("KWS model load 끝.")


''' 10s 녹음 후 저장 하는 소스코드'''
FILE_NAME = "dev_file"
RECORD_FILE_PATH = record(FILE_NAME)

## others
#RECORD_FILE_PATH = './record_data/연구필요/10s_성공_stop_seven_cat.wav'
#RECORD_FILE_PATH = './record_data/연구필요/10s_one_stop_two.wav'  # peack_detect 오류
#RECORD_FILE_PATH = './record_data/연구필요/up_high_probability.wav'

print("ACTIVATE KEYWORD 인식 중......")
ex_FILE_PATH = KWS(RECORD_FILE_PATH, KWS_model)

threshold, other_words = activation(ex_FILE_PATH)
print("stop probability for threshold:", threshold)  # 천천히 말하면 2음절로 분리되어 up으로 인식함.. >> 빠르게 말해
print("up probabilty:", other_words[26])
print("other:", other_words)

if threshold > 0.6 : #and other_words[26]:
    print("there is stop word, ACTIVATE")
    sentence = STT(RECORD_FILE_PATH)
    print(sentence)

else:
    print("NOT ACTIVATE")