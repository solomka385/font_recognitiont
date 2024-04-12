import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import nltk
from nltk.corpus import brown
from collections import Counter
import re
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

nltk.download('brown')

words = brown.words()


word_freq = Counter(words)

# нейронная сеть обучалась на выборке из 12500 данных, для того, чтобы выложить на гитхаб была создана отдельная выборка на более меньшем наборе
popular_words = [word for word, _ in word_freq.most_common(10)]


resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224)) 
    image = np.array(image)  
    image = preprocess_input(image)  
    image = np.expand_dims(image, axis=0)  
    return image

def generate_images(font_file, output_dir, words, font_name, df_font):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
   
    font = ImageFont.truetype(font_file, size=50)
    
    
    for word in words:
        
        filename = re.sub(r'[\\/*?:"<>|]', '_', word)
        
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), word, fill="black", font=font)
        image.save(os.path.join(output_dir, f'{filename}.png'), "PNG")
        
        features = resnet_model.predict(load_and_preprocess_image(os.path.join(output_dir, f'{filename}.png')))
        
        df_font.loc[len(df_font)] = [features.tolist(), font_name]  


font_folder = "fonts/"



df_font = pd.DataFrame(columns=['vect', 'font'])

font_folder = "fonts/"
output_base_dir = "img_front/"

for root, dirs, files in os.walk(font_folder):
    for font_file in files:
        if font_file.endswith(".ttf") or font_file.endswith(".otf"):
            font_name = os.path.splitext(font_file)[0]
            output_dir = os.path.join(output_base_dir, "", font_name)
            font_file_path = os.path.join(root, font_file)
            generate_images(font_file_path, output_dir, popular_words, font_name, df_font)

df_font.to_csv('image_datasetq.csv', index=False)

