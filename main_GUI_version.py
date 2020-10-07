# PyQt5랑 playsound 깔아야함!!

''' import libraries '''
from inference import *

classes = ['bed', 'bird', 'cat', 'dog', 'down',
           'eight', 'five', 'four', 'go', 'happy',
           'house', 'left', 'marvin', 'nine', 'no',
           'off', 'on', 'one', 'right', 'seven',
           'sheila', 'six', 'stop', 'three', 'tree',
           'two', 'up', 'wow', 'yes', 'zero']



## GUI

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5 import QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QPixmap

from PyQt5 import QtCore, QtWidgets, QtMultimedia

from wav_utils import *

from playsound import playsound

from PyQt5.QtGui import QPixmap


class MyApp(QWidget):
    def __init__(self, KWS_model):
        super().__init__()

        self.initUI()
        # default
        self.RECORD_FILE_PATH = "record_data/연구필요/10s_성공_stop_seven_cat.wav"
        self.ex_FILE_PATH = None
        self.threshold, self.other_words = None, None
        self.KWS_model = KWS_model

    def initUI(self):

        # 버튼
        btn1 = QPushButton(self)
        btn1.setText('녹음 버튼')  # & Keyword 인식
        btn1.clicked.connect(self.on_click1)
        btn1.move(20, 20)

        btn2 = QPushButton(self)
        btn2.setText('녹음 파일 듣기')
        btn2.clicked.connect(self.on_click2)
        btn2.move(20, 60)

        btn3 = QPushButton(self)
        btn3.setText('STT 결과 확인')
        btn3.clicked.connect(self.on_click3)
        btn3.move(20, 100)

        # x 초 녹음 중 출력 예정
        self.STT_result = QLabel(self)
        self.Keyword_result = QLabel(self)

        # 이미지 업로드
        self.pixmap1 = QPixmap()
        self.pixmap2 = QPixmap()
        self.lbl_img1 = QLabel()
        self.lbl_img1.setAlignment(Qt.AlignHCenter)
        self.lbl_img2 = QLabel()
        self.lbl_img2.setAlignment(Qt.AlignHCenter)


        # layout 설정
        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(btn1)
        hbox1.addWidget(btn2)
        hbox1.addStretch(1)


        vbox = QVBoxLayout()
        vbox.addStretch(2)
        vbox.addLayout(hbox1)
        vbox.addWidget(self.Keyword_result)
        vbox.addWidget(self.STT_result)
        vbox.addWidget(btn3)
        vbox.addWidget(self.lbl_img1)
        vbox.addWidget(self.lbl_img2)

        # vbox = QVBoxLayout()
        # vbox.addWidget(btn1)
        # vbox.addWidget(btn2)
        # vbox.addWidget(btn3)
        #
        # vbox.addWidget(self.Keyword_result)
        # vbox.addWidget(self.STT_result)
        # vbox.addWidget(self.lbl_img1)
        # vbox.addWidget(self.lbl_img2)

        # 창 띄우는 부분
        self.setLayout(vbox)
        self.setWindowTitle('[Demo] Limited Vocabulary KWS & Classifier for STT')
        self.setGeometry(300, 300, 400, 200)
        self.show()

    def on_click1(self):

        # print("자동실행 버튼을 클릭 하셨군요")

        print("record start!")
        #self.STT_result.setText("record start!")
        FILE_NAME = "dev_file"
        self.RECORD_FILE_PATH = record(FILE_NAME)

        QMessageBox.about(self, "message", "녹음 완료")
        #return WAVE_OUTPUT_FILENAME

        print("ACTIVATE KEYWORD 인식 중......")
        print("record_file", self.RECORD_FILE_PATH)
        self.ex_FILE_PATH = KWS(self.RECORD_FILE_PATH, self.KWS_model)
        self.threshold, self.other_words = activation(self.ex_FILE_PATH)


    def on_click2(self):

        playsound(self.RECORD_FILE_PATH)  # sound 먼저


    def on_click3(self):

        if self.ex_FILE_PATH == None:
            print("계산 중..")
            print("ACTIVATE KEYWORD 인식 중......")
            print("record_file", self.RECORD_FILE_PATH)
            self.ex_FILE_PATH = KWS(self.RECORD_FILE_PATH, self.KWS_model)
            self.threshold, self.other_words = activation(self.ex_FILE_PATH)


        if self.threshold > 0.6:  # and other_words[26]:
            print("there is stop word, ACTIVATE")
            self.Keyword_result.setText("Keyword 있음")
            sentence_list = STT(self.RECORD_FILE_PATH)
            sentence = ' '.join(sentence_list)
            print(sentence)
            self.STT_result.setText(sentence)

            self.pixmap1.load("dev_output/KWS_Keyword_Position.jpg")
            self.pixmap2.load("dev_output/STT_extraction_position.jpg")

            self.lbl_img1.setPixmap(self.pixmap1)
            self.lbl_img2.setPixmap(self.pixmap2)

        else:
            self.Keyword_result.setText("Keyword 없음")
            print("NOT ACTIVATE")






if __name__ == '__main__':

    # KWS 모델을 먼저 불러볼까?
    print("KWS model load 중 ......")
    KWS_model_name = "./model/KWS/KWS_GRU_2layer"

    # model, optimizer 초기화
    KWS_model = KWS_2GRU()  # .to(device)
    KWS_model_path = KWS_model_name + '_ckp.tar'
    checkpoint = torch.load(KWS_model_path, map_location=device)
    KWS_model.load_state_dict(checkpoint['model_state_dict'])
    print("KWS model load 끝.")

    app = QApplication(sys.argv)
    ex = MyApp(KWS_model)
    sys.exit(app.exec_())
    
    os.system('pause')  # exe 파일 바로 꺼지는거 방지