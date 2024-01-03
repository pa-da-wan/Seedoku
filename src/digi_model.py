import numpy as np
import tensorflow as tf
import cv2

class DigitRecognitionModel:
    def __init__(self, model_path):
        self.loaded_model = tf.keras.models.load_model(model_path)

    def preprocess_cell(self, cell):
        digit_img = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        digit_img = digit_img[5:-5, 5:-5]
        digit_img = cv2.resize(digit_img, (28, 28))
        digit_img = digit_img.reshape((28, 28, 1))  
        return digit_img

    def predict_digit(self, digit_img):
        predictions = self.loaded_model.predict(digit_img, verbose=0)
        return np.argmax(predictions), np.max(predictions)

    def get_grid_numbers(self, cells):
        cell_images = np.array([self.preprocess_cell(cell) for cell in cells])
        predictions = self.loaded_model.predict(cell_images, verbose=0)

        threshold = 0.75
        predicted_digits = np.argmax(predictions, axis=1) * (np.max(predictions, axis=1) > threshold)
        return predicted_digits.reshape((9, 9))
