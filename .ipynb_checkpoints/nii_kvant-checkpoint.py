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

# Загрузка корпуса Brown
nltk.download('brown')

# Получение списка слов из корпуса Brown
words = brown.words()

# Подсчет частоты каждого слова
word_freq = Counter(words)

# Получение 5000 самых популярных слов
popular_words = [word for word, _ in word_freq.most_common(1000)]

# Инициализация предобученной модели ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Размер, ожидаемый ResNet50
    image = np.array(image)  # Преобразуем изображение в массив numpy
    image = preprocess_input(image)  # Предобработка изображения для ResNet50
    image = np.expand_dims(image, axis=0)  # Добавляем размерность пакета
    return image

# Функция для создания изображений символов и слов из шрифта
def generate_images(font_file, output_dir, words, font_name, df_font):
    # Создаем папку для сохранения изображений, если ее еще нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загружаем шрифт
    font = ImageFont.truetype(font_file, size=50)
    
    # Генерируем изображения для каждого слова
    for word in words:
        # Заменяем недопустимые символы в имени файла
        filename = re.sub(r'[\\/*?:"<>|]', '_', word)
        
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), word, fill="black", font=font)
        image.save(os.path.join(output_dir, f'{filename}.png'), "PNG")
        # Извлекаем признаки изображения с помощью ResNet50
        features = resnet_model.predict(load_and_preprocess_image(os.path.join(output_dir, f'{filename}.png')))
        # Добавляем признаки и название шрифта в DataFrame
        df_font.loc[len(df_font)] = [features.tolist(), font_name]  # Преобразуем массив NumPy в список

# Путь к папке с шрифтами
font_folder = "fonts/"


# Создаем DataFrame для хранения признаков и шрифтов
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

# Сохраняем DataFrame в CSV файл
df_font.to_csv('image_datasetq.csv', index=False)

