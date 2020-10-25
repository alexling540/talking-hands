from PyQt5 import QtCore
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


class TF_Worker(QtCore.QThread):
    data = QtCore.pyqtSlot(np.array)

    def __init__(self, labels, image_name, parent=None):
        super(TF_Worker, self).__init__(parent)
        self._stopped = True
        self._mutex = QtCore.QMutex()

        self.graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile('model/model.pb', 'rb') as f:
            self.graph_def.ParseFromString(f.read())
            tf.import_graph_def(self.graph_def, name='')

        self.labels = []
        with open('model/labels.txt', 'rb') as lf:
            for l in lf:
                self.labels.append(l.strip())

        self.image_name = image_name

    def stop(self):
        self._mutex.lock()
        self._stopped = True
        self._mutex.unlock()

    def run(self):
        self._stopped = False

        image = Image.open(self.image_name)
        image = update_orientation(image)
        image = convert_to_opencv(image)
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

        res = []

        for index, p in enumerate(predictions):
            truncated_probability = np.float64(np.round(p, 8))
            print(self.labels[index], truncated_probability)
            self.res.append((self.labels[index], truncated_probability))

        res = np.array(res)
        self.data.emit(res)


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
