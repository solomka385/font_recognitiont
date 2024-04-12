# Укажите базовый образ с Python
FROM python:3.9

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Установите модель и другие служебные файлы
COPY my_model.h5 /
COPY neural_ux.py /

# Задайте рабочую директорию
WORKDIR /


# Укажите команду для запуска приложения
CMD ["python", "neural_ux.py"]
