from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from datetime import datetime
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from msrest.authentication import ApiKeyCredentials


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

        btn_layout = QVBoxLayout()

        cam_open_close_layout = QHBoxLayout()

        cam_open = QPushButton('Open cam')
        cam_open.clicked.connect(self.openCamera)
        cam_open_close_layout.addWidget(cam_open)

        cam_close = QPushButton('Close cam')
        cam_close.clicked.connect(self.stopCamera)
        cam_open_close_layout.addWidget(cam_close)

        btn_layout.addLayout(cam_open_close_layout)

        cam_ss = QPushButton('Take Screenshot')
        cam_ss.clicked.connect(self.saveScreen)
        btn_layout.addWidget(cam_ss)

        layout.addLayout(btn_layout)

        self.returnedString = QLineEdit("")
        layout.addWidget(self.returnedString)

        self.videoFeed = QLabel()
        self.videoFeed.setFixedSize(640, 480)
        layout.addWidget(self.videoFeed)

        self.setLayout(layout)
        self.setWindowTitle("Talking Hands")
        self.setFixedSize(800, 800)

    def openCamera(self):
        self.vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # windows only fix
        # vc.set(5, 30)  #set FPS
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        if not self.vc.isOpened():
            print('unable to open camera feed')
            return

        self.timer.start(1000. / 24)

    def stopCamera(self):
        self.videoFeed.setPixmap(QPixmap())
        self.frame = None
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
            ts = datetime.now().timestamp()
            image_name = "{}.jpg".format(ts)
            # success = cv2.imwrite(image_name, self.frame[200:450, 50:300])
            # print(success)

            image = self.frame[200:450, 50:300]
            # tf_worker = TF_Worker(labels=[], imge_name=image_name)
            # tf_worker.data.connect(self.worker_callback)
            # tf_worker.start()

            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile('model/model.pb', 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            labels = []
            with open('model/labels.txt', 'rb') as lf:
                for l in lf:
                    labels.append(l.strip())
            print(labels)

            # image = Image.open(image_name)
            # image = update_orientation(image)
            # image = convert_to_opencv(image)
            image = resize_down_to_1600_max_dim(image)

            h, w = image.shape[:2]
            min_dim = min(w, h)
            max_square_image = crop_center(image, min_dim, min_dim)

            augmented_image = resize_to_256_square(max_square_image)

            with tf.compat.v1.Session() as sess:
                input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
            network_input_size = input_tensor_shape[1]

            # Crop the center for the specified network_input_Size
            augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

            with tf.compat.v1.Session() as sess:
                try:
                    prob_tensor = sess.graph.get_tensor_by_name('loss:0')
                    predictions = sess.run(prob_tensor, {'Placeholder:0': [augmented_image]})
                except KeyError:
                    print("error")

            print(predictions)
            print(np.argmax(predictions))

            highest_probability_index = np.argmax(predictions)
            pred_label = labels[highest_probability_index].decode("utf-8")
            print('Classified as: ' + pred_label)

            self.returnedString.setText(self.returnedString.text() + pred_label)

            #
            # for index, p in enumerate(predictions):
            #     truncated_probability = np.float64(np.round(p, 8))
            #     print(labels[index], truncated_probability)

        else:
            print('x')

            # r = requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction' +
            #                   '/e58f5c9f-c12f-49c2-81f2-5544d2879949/classify/iterations/Iteration1/url',
            #                   headers={
            #                       "Prediction-Key": "405401075ae14cabaff7de7e18880d23",
            #                       "Content-Type": "application/octet-stream"
            #                   },
            #                   files={'image': open("{}.jpg".format(ts), 'rb')})
            # json = r.json()
            # print(r.status_code, json)
            # self.returnedString.setText(self.returnedString.text() + 'x')

    def worker_callback(self, data):
        print(data)


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r, g, b = np.array(image).T
    opencv_image = np.array([b, g, r]).transpose()
    return opencv_image


def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if h < 1600 and w < 1600:
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif != None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
