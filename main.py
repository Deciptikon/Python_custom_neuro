import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import torch
print(f'Cuda is {torch.cuda.is_available()}')

# Функция для загрузки и обработки изображения с использованием нейросети
def process_image():
    # Загрузка изображения
    file_path = filedialog.askopenfilename()
    original_image = Image.open(file_path)
    
    # Преобразование изображения к размеру (64, 64)
    resized_image = original_image.resize((64, 64))
    
    # Преобразование изображения в массив NumPy
    image_array = np.array(resized_image) / 255.0  # Нормализация значений пикселей
    
    # Добавление дополнительной размерности для соответствия ожидаемому входу сети
    image_array = np.expand_dims(image_array, axis=0)
    
    # Получение предсказания сети
    prediction = 226 #model.predict(image_array)
    
    # Отображение оригинального изображения
    original_img_label.img = ImageTk.PhotoImage(original_image)
    original_img_label.config(image=original_img_label.img)
    
    # Отображение результата работы сети
    result_label.config(text=f'Prediction: {prediction}')#{prediction[0, 0]:.2f}

# Создание основного окна Tkinter
root = tk.Tk()
root.title("Image Processing with Neural Network")
root.geometry("1000x600+400+200")

# Кнопка для выбора изображения
load_button = tk.Button(root, text="Load Image", command=process_image)
load_button.pack(pady=10)

# Метка для отображения оригинального изображения
original_img_label = tk.Label(root)
original_img_label.pack()

# Метка для отображения результата работы сети
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Запуск Tkinter-приложения
root.mainloop()