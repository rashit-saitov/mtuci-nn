import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt


# Генерация бинарного представления символа
def generate_symbol_image(symbol, font_path, image_size=(20, 20)):
    image = Image.new('L', image_size, color=255)
    draw = ImageDraw.Draw(image)

    if not os.path.isfile(font_path):
        raise ValueError(f"Файл шрифта '{font_path}' не найден.")

    font = ImageFont.truetype(font_path, 15)
    text_bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(text_position, symbol, fill=0, font=font)
    binary_image = np.array(image).flatten() // 255
    return binary_image

# Пример генерации и визуализации символа
def visualize_symbol(symbol, font_path):
    binary_image = generate_symbol_image(symbol, font_path)
    image = binary_image.reshape((20, 20))
    plt.imshow(image, cmap='gray')
    plt.title(symbol)
    plt.show()


# Список символов
symbols = ['A', 'B', 'C', 'D']

fonts_directory = "/Users/rmsaitov/wrs/mtuci/nn/lab1"
fonts = [
    os.path.join(fonts_directory, 'Arial.ttf'),
    os.path.join(fonts_directory, 'Verdana.ttf'),
    os.path.join(fonts_directory, 'TimesNewRoman.ttf'),
    os.path.join(fonts_directory, 'LiberationSerif-Regular.ttf')
]

# Проверяем наличие файлов шрифтов
for font in fonts:
    if not os.path.isfile(font):
        print(f"Ошибка: файл шрифта '{font}' не найден. Укажите правильный путь к шрифту.")
        exit(1)

training_data = []
outputs = []

# Создаем обучающую выборку
for i, symbol in enumerate(symbols):
    for font in fonts:
        binary_image = generate_symbol_image(symbol, font)
        training_data.append(binary_image)
        output = [0] * len(symbols)
        output[i] = 1
        outputs.append(output)

training_data = np.array(training_data)
outputs = np.array(outputs)

# Инициализация весов
np.random.seed(0)
n_inputs = len(training_data[0])
n_outputs = len(symbols)
weights = np.random.randn(n_inputs, n_outputs)
learning_rate = 0.1
epochs = 10000


def activation(x):
    return 1 / (1 + np.exp(-x))  # Сигмоида


def predict(inputs):
    return activation(np.dot(inputs, weights))


# Алгоритм обучения с использованием однослойного персептрона и градиентного спуска
for epoch in range(epochs):
    for inputs, target in zip(training_data, outputs):
        # Прямой проход
        outputs_pred = predict(inputs)

        # Ошибка
        error = target - outputs_pred

        # Обновление весов
        weights += learning_rate * np.dot(inputs[:, None], error[None, :])

    if epoch % 1000 == 0:
        total_error = np.sum(np.square(outputs - predict(training_data)))
        print(f'Epoch {epoch}, Total Error: {total_error}')

print("Обучение завершено")

visualize_symbol('A', os.path.join(fonts_directory, 'Arial.ttf'))
visualize_symbol('B', os.path.join(fonts_directory, 'Verdana.ttf'))
visualize_symbol('C', os.path.join(fonts_directory, 'TimesNewRoman.ttf'))
visualize_symbol('D', os.path.join(fonts_directory, 'LiberationSerif-Regular.ttf'))

# Тестирование на новых данных
test_font = os.path.join(fonts_directory, 'Arial.ttf')  # Шрифт, который не использовался в обучающей выборке
test_data = [generate_symbol_image(symbol, test_font) for symbol in symbols]

for test_case, symbol in zip(test_data, symbols):
    output = predict(test_case)
    predicted_symbol_index = np.argmax(output)
    predicted_symbol = symbols[predicted_symbol_index]
    print(f'Test case: {symbol}, Predicted: {predicted_symbol}')
