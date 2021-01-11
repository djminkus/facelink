# coding:utf-8

# FaceLink
# A social networking app for 2021
import os

from kivy.app import App
# from kivy.uix.image import Image
import kivy.uix.image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers.merge import Concatenate
from keras import backend as K

from PIL import Image

# OpenFace
dump = False  # Dump recognition output stuff into console?
color = (67, 67, 67)

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['David', 'Hakan', 'Unknown']
bios = [
    '''seeking Master's in Electrical Engineering at Colorado School of Mines''',
    'seeking a Ph.D. at Colorado School of Mines',
    'this person is not a FaceLink user.'
]

# ---- everything before cam loop:
counter = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('trainer/trained_model.yml')

# DNN Stuff:
# Define paths for DNN
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')


# OpenFace stuff
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(96, 96))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    # Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img


def buildModel():
    myInput = Input(shape=(96, 96, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = Lambda(lambda x: x ** 2, name='power2_3b')(inception_3a)
    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name='inception_3c_3x3_conv1')(inception_3b)
    inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn1')(inception_3c_3x3)
    inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
    inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
    inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_3c_3x3_conv' + '2')(inception_3c_3x3)
    inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn' + '2')(inception_3c_3x3)
    inception_3c_3x3 = Activation('relu')(inception_3c_3x3)

    inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_3c_5x5_conv1')(inception_3b)
    inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn1')(inception_3c_5x5)
    inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
    inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
    inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name='inception_3c_5x5_conv' + '2')(inception_3c_5x5)
    inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn' + '2')(inception_3c_5x5)
    inception_3c_5x5 = Activation('relu')(inception_3c_5x5)

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    # inception 4a
    inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_4a_3x3_conv' + '1')(inception_3c)
    inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn' + '1')(inception_4a_3x3)
    inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
    inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
    inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name='inception_4a_3x3_conv' + '2')(inception_4a_3x3)
    inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn' + '2')(inception_4a_3x3)
    inception_4a_3x3 = Activation('relu')(inception_4a_3x3)

    inception_4a_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_4a_5x5_conv1')(inception_3c)
    inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn1')(inception_4a_5x5)
    inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
    inception_4a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
    inception_4a_5x5 = Conv2D(64, (5, 5), strides=(1, 1), name='inception_4a_5x5_conv' + '2')(inception_4a_5x5)
    inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn' + '2')(inception_4a_5x5)
    inception_4a_5x5 = Activation('relu')(inception_4a_5x5)

    inception_4a_pool = Lambda(lambda x: x ** 2, name='power2_4a')(inception_3c)
    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)

    inception_4a_pool = Conv2D(128, (1, 1), strides=(1, 1), name='inception_4a_pool_conv' + '')(inception_4a_pool)
    inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_pool_bn' + '')(inception_4a_pool)
    inception_4a_pool = Activation('relu')(inception_4a_pool)
    inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

    inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_4a_1x1_conv' + '')(inception_3c)
    inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_1x1_bn' + '')(inception_4a_1x1)
    inception_4a_1x1 = Activation('relu')(inception_4a_1x1)

    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    # inception4e
    inception_4e_3x3 = Conv2D(160, (1, 1), strides=(1, 1), name='inception_4e_3x3_conv' + '1')(inception_4a)
    inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn' + '1')(inception_4e_3x3)
    inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
    inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
    inception_4e_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_4e_3x3_conv' + '2')(inception_4e_3x3)
    inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn' + '2')(inception_4e_3x3)
    inception_4e_3x3 = Activation('relu')(inception_4e_3x3)

    inception_4e_5x5 = Conv2D(64, (1, 1), strides=(1, 1), name='inception_4e_5x5_conv' + '1')(inception_4a)
    inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn' + '1')(inception_4e_5x5)
    inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
    inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
    inception_4e_5x5 = Conv2D(128, (5, 5), strides=(2, 2), name='inception_4e_5x5_conv' + '2')(inception_4e_5x5)
    inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn' + '2')(inception_4e_5x5)
    inception_4e_5x5 = Activation('relu')(inception_4e_5x5)

    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

    # inception5a
    inception_5a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_5a_3x3_conv' + '1')(inception_4e)
    inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn' + '1')(inception_5a_3x3)
    inception_5a_3x3 = Activation('relu')(inception_5a_3x3)
    inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
    inception_5a_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name='inception_5a_3x3_conv' + '2')(inception_5a_3x3)
    inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn' + '2')(inception_5a_3x3)
    inception_5a_3x3 = Activation('relu')(inception_5a_3x3)

    inception_5a_pool = Lambda(lambda x: x ** 2, name='power2_5a')(inception_4e)
    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)

    inception_5a_pool = Conv2D(96, (1, 1), strides=(1, 1), name='inception_5a_pool_conv' + '')(inception_5a_pool)
    inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_pool_bn' + '')(inception_5a_pool)
    inception_5a_pool = Activation('relu')(inception_5a_pool)
    inception_5a_pool = ZeroPadding2D(padding=(1, 1))(inception_5a_pool)

    inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_5a_1x1_conv' + '')(inception_4e)
    inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_1x1_bn' + '')(inception_5a_1x1)
    inception_5a_1x1 = Activation('relu')(inception_5a_1x1)

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    # inception_5b
    inception_5b_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_5b_3x3_conv' + '1')(inception_5a)
    inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn' + '1')(inception_5b_3x3)
    inception_5b_3x3 = Activation('relu')(inception_5b_3x3)
    inception_5b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
    inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name='inception_5b_3x3_conv' + '2')(inception_5b_3x3)
    inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn' + '2')(inception_5b_3x3)
    inception_5b_3x3 = Activation('relu')(inception_5b_3x3)

    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

    inception_5b_pool = Conv2D(96, (1, 1), strides=(1, 1), name='inception_5b_pool_conv' + '')(inception_5b_pool)
    inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_pool_bn' + '')(inception_5b_pool)
    inception_5b_pool = Activation('relu')(inception_5b_pool)

    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_5b_1x1_conv' + '')(inception_5a)
    inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_1x1_bn' + '')(inception_5b_1x1)
    inception_5b_1x1 = Activation('relu')(inception_5b_1x1)

    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

    # Final Model
    model = Model(inputs=[myInput], outputs=norm_layer)
    return model


of_model = buildModel()  # OpenFace model
print("model built")

# https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view
of_model.load_weights('openface_weights.h5')
print("weights loaded")


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    # euclidean_distance = l2_normalize(euclidean_distance)
    return euclidean_distance


metric = "cosine"  # cosine, euclidean

if metric == "cosine":
    threshold = 0.45
else:
    threshold = 0.95

# Put your employee (etc.) pictures in this path as name_of_employee.jpg
# employee_pictures = "database/"
user_pictures = "user_faces/"

# employees = dict()
users = dict()

for file in os.listdir(user_pictures):  # Fill dictionary with image representations
    user, extension = file.split(".")
    img = preprocess_image('user_faces/%s.jpg' % user)  # *further changes here?
    representation = of_model.predict(img)[0, :]

    users[user] = representation

print("employee representations retrieved successfully")


# Kivy setup (function to process each frame is here):
class KivyCamera(kivy.uix.image.Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture  # defined as cv2.VideoCapture(0) in CamApp build func
        Clock.schedule_interval(self.update, 1.0 / fps)

        # self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.fd_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        self.window = ['a', 'b', 'c', 'd', 'e']  # 5 most recent usernames

        # self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        # self.face_recognizer.read('trainer/Trained_Model_w_DNN_hakan.yml')

    def update(self, dt):
        _, frame = self.capture.read()  # Frame is image
        if _:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(gray)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            self.fd_model.setInput(blob)
            detections = self.fd_model.forward()

            count = 0
            faces = []
            # Identify each face
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                # If confidence > 0.5, do stuff
                if confidence > 0.5:
                    count += 1
                    face = frame[startY:endY, startX:endX]
                      # ^ ***
                    # cv2.imwrite(base_dir + '/dnn_extracted_faces/' + str(i) + '_' + file, frame_)
                    faces.append(face)

                    h = endY - startY
                    w = endX - startX
                    x = startX
                    y = startY

                    detected_face = cv2.resize(face, (96, 96))  # resize to 96x96

                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    # user dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                    img_pixels /= 127.5
                    img_pixels -= 1

                    captured_representation = of_model.predict(img_pixels)[0, :]

                    distances = []

                    for u in users:  # For each user, find "distance" between detected face and that user's face
                        user_name = u
                        source_representation = users[u]

                        if metric == "cosine":
                            distance = findCosineDistance(captured_representation, source_representation)
                        elif metric == "euclidean":
                            distance = findEuclideanDistance(captured_representation, source_representation)

                        if dump:
                            print(user_name, ": ", distance)
                        distances.append(distance)

                    label_name = 'unknown'
                    color = (100, 100, 100)
                    similarity = 0
                    index = 0
                    for u in users:  #
                        user_name = u
                        if index == np.argmin(distances):  # If this index is that of the minimum val in distances...
                            if distances[index] <= threshold:  # and that distance is less than the chosen threshold...
                                # print("detected: ",user_name)

                                if metric == "euclidean":
                                    similarity = 100 + (90 - 100 * distance)
                                elif metric == "cosine":
                                    similarity = 100 + (40 - 100 * distance)

                                if similarity > 99.99: similarity = 99.99

                                # label_name = "%s (%s%s)" % (user_name, str(round(similarity, 2)), '%')

                                # sim_str = "  {0}%".format(round(similarity))

                                # Color-code box based on similarity level:
                                if similarity > 80:
                                    color = (0, 255, 0)  # green box
                                elif similarity > 60:
                                    color = (0, 255, 255)  # yellow box
                                else:
                                    color = (0, 0, 255)  # red box

                                break  # User match found; stop checking.

                        index = index + 1

                    # id, inv_conf = self.face_recognizer.predict(gray[y:y + h, x:x + w])
                    # if inv_conf < 100:
                    #     id = names[id - 1]
                    #     confidence = "  {0}%".format(round(100 - inv_conf))
                    # else:
                    #     id = "unknown"
                    #     confidence = "  {0}%".format(round(100 - inv_conf))

                    # According the similarity results face box will be change
                    # if inv_conf < 20:
                    #     color = (0, 255, 0)  # green box
                    # elif inv_conf < 40:
                    #     color = (0, 255, 255)  # yellow box
                    # else:
                    #     color = (0, 0, 255)  # red box

                    if similarity is 0:
                        user_name = 'unknown'

                    self.window.append(user_name)
                    # Find "mode" of window

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frame, str(user_name), (x + 5, y - 5), font, 1, color, 2)
                    cv2.putText(frame, str(similarity), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)



            # OpenFace stuff: ----------------------------------------------
            #   Given "faces" from Haar detector (not our "faces"):
            # for (x, y, w, h) in faces:
            #     if w > 130:  # discard small detected faces
            #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)  # draw rectangle to main image
            #
            #         detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            #           # ^ ***
            #         detected_face = cv2.resize(detected_face, (96, 96))  # resize to 96x96
            #
            #         img_pixels = image.img_to_array(detected_face)
            #         img_pixels = np.expand_dims(img_pixels, axis=0)
            #         # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
            #         img_pixels /= 127.5
            #         img_pixels -= 1
            #
            #         captured_representation = of_model.predict(img_pixels)[0, :]
            #
            #         distances = []
            #
            #         for i in users:
            #             user_name = i
            #             source_representation = users[i]
            #
            #             if metric == "cosine":
            #                 distance = findCosineDistance(captured_representation, source_representation)
            #             elif metric == "euclidean":
            #                 distance = findEuclideanDistance(captured_representation, source_representation)
            #
            #             if dump:
            #                 print(user_name, ": ", distance)
            #             distances.append(distance)
            #
            #         label_name = 'unknown'
            #         index = 0
            #         for i in users:
            #             user_name = i
            #             if index == np.argmin(distances):
            #                 if distances[index] <= threshold:
            #                     # print("detected: ",user_name)
            #
            #                     if metric == "euclidean":
            #                         similarity = 100 + (90 - 100 * distance)
            #                     elif metric == "cosine":
            #                         similarity = 100 + (40 - 100 * distance)
            #
            #                     if similarity > 99.99: similarity = 99.99
            #
            #                     label_name = "%s (%s%s)" % (user_name, str(round(similarity, 2)), '%')
            #
            #                     break
            #
            #             index = index + 1
            #
            #         cv2.putText(img, label_name, (int(x + w + 15), int(y - 64)), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                     (255, 255, 255), 2)
            #
            #         if dump:
            #             print("----------------------")
            #
            #         # connect face and text
            #         cv2.line(img, (x + w, y - 64), (x + w - 25, y - 64), color, 1)
            #         cv2.line(img, (int(x + w / 2), y), (x + w - 25, y - 64), color, 1)

            # END OpenFace stuff --------------------------------------------------------------------------

            # convert it to texture for display in app:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        # Without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()
