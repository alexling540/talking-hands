from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from datetime import datetime
import cv2


class UI_Window(QWidget):

    def __init__(self):
        super().__init__()

        self.vc = None
        self.frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)

        layout = QVBoxLayout()

        label = QLabel("Hello World")
        layout.addWidget(label)

        btnCam = QPushButton('Open cam')
        btnCam.clicked.connect(self.openCamera)
        layout.addWidget(btnCam)

        btnCam2 = QPushButton('Close cam')
        btnCam2.clicked.connect(self.stopCamera)
        layout.addWidget(btnCam2)

        btnCam3 = QPushButton('Take Screenshot')
        btnCam3.clicked.connect(self.saveScreen)
        layout.addWidget(btnCam3)

        self.videoFeed = QLabel()
        self.videoFeed.setFixedSize(640, 640)
        layout.addWidget(self.videoFeed)

        self.setLayout(layout)
        self.setWindowTitle("Talking Hands")
        self.setFixedSize(800, 800)

    def openCamera(self):
        self.vc = cv2.VideoCapture(0, cv2.CAP_DSHOW) # windows only fix
        # vc.set(5, 30)  #set FPS
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        if not self.vc.isOpened():
            print('unable to open camera feed')
            return

        self.timer.start(1000. / 24)

    def stopCamera(self):
        self.timer.stop()

    def nextFrameSlot(self):
        rval, frame = self.vc.read()
        if frame is not None:
            self.frame = frame

            n_frame = cv2.rectangle(frame, (50, 200), (300, 450), (0, 255, 0), 5)

            n_frame = cv2.cvtColor(n_frame, cv2.COLOR_BGR2RGB)
            image = QImage(n_frame, n_frame.shape[1], n_frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.videoFeed.setPixmap(pixmap)

    def saveScreen(self):
        if self.frame is not None:
            success = cv2.imwrite("{}.jpg".format(datetime.now().timestamp()), self.frame[200:450, 50:300])
            print(success)
