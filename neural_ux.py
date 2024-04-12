from keras.models import load_model
from PIL import Image
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
class predict_img():
    def __init__(self):
        self.model = load_model('my_model.h5')
        self.label_encoder = LabelEncoder()
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224)) 
        image = np.array(image)  
        image = preprocess_input(image)  
        image = np.expand_dims(image, axis=0)  
        return image
    
    def check_class(self, path_to_img):
        features = self.resnet_model.predict(self.load_and_preprocess_image(path_to_img))
        predicted_class_probs = self.model.predict(features)
        predicted_class_index = np.argmax(predicted_class_probs)
        probability = predicted_class_probs[0][predicted_class_index]  # Получение вероятности класса
        return "Новое фото отнесено к категории: {}, с вероятностью: {:.2f}".format(predicted_class_index, probability)
import time
#path_to_imgg = 'test_new_img/alumni_italic.jpg'
if __name__ == "__main__":
    path_to_img = input('Введите путь до файла: ')
    
    img = predict_img()
    print(img.check_class(path_to_img))
    
    