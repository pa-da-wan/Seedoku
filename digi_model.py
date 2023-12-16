import numpy as np
import tensorflow as tf
import cv2

class DigitRecognitionModel:
    def __init__(self, model_path):
        self.loaded_model = tf.keras.models.load_model(model_path)

    def prepareCell(self, cell):
        digit_img = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        digit_img = digit_img[5:-5, 5:-5]
        digit_img = cv2.resize(digit_img, (28, 28))
        digit_img = digit_img.reshape((1, 28, 28, 1))
        return digit_img

    def predictDigit(self, digit_img):
        prediction = np.argmax(self.loaded_model.predict(digit_img, verbose=0))
        prob = np.max(self.loaded_model.predict(digit_img, verbose=0))
        return prediction, prob
