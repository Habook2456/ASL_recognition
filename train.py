import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import CNN
import matplotlib.pyplot as plt

IMG_SIZE = 300
LR = 1e-3
CLASS_LABELS = [chr(letra_ascii) for letra_ascii in range(ord('A'), ord('Z')+1)]
nb_classes = len(CLASS_LABELS)
DATA_DIR = 'Data'
MODEL_NAME = 'handsign.model'


def load_data():
    X = []
    Y = []
    for i, class_label in enumerate(CLASS_LABELS):
        class_dir = os.path.join(DATA_DIR, class_label)
        file_list = os.listdir(class_dir)
        for file_name in file_list:
            file_path = os.path.join(class_dir, file_name)
            image = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            X.append(image)
            Y.append(i)
    X = np.array(X)
    Y = to_categorical(Y, nb_classes)
    return X, Y

X, Y = load_data()

X = X/255

model = CNN.vgg16_model()

model.fit(X, Y, epochs=25, validation_split=0.1)

model.save(MODEL_NAME)
print("Modelo entrenado y guardado como '{}'".format(MODEL_NAME))