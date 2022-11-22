import numpy as np
import cv2 as cv
# обучение
image20 = 'images/20.png' # Знак 20
image30 = 'images/30.png' # Знак 30
image60 = 'images/60.png' # Знак 60
# тестовые изображения
image20test = 'images/20.png' # Знак 20
image30test = 'images/30.png' # Знак 30
image60test = 'images/60.png' # Знак 60
image50test = 'images/50.png' # Знак 50
image20_5test = 'images/20.5.png' # Знак 20 на 5 градусов
image20_10test = 'images/20.10.png' # Знак 20 на 10 градусов
image20_15test = 'images/20.15.png' # Знак 20 на 15 градусов
# делаем из изображений массив
def img_to_arr(image):
    array = cv.imread(image, 0).astype(float)
    return array
# делаем массивы для обучения
my_array = np.array([
    img_to_arr(image20).flatten(),
    img_to_arr(image30).flatten(),
    img_to_arr(image60).flatten()],)
# делаем массивы для теста
test_array = np.array([
    img_to_arr(image20test).flatten(),
    img_to_arr(image30test).flatten(),
    img_to_arr(image60test).flatten(),
    img_to_arr(image50test).flatten(),
    img_to_arr(image20_5test).flatten(),
    img_to_arr(image20_10test).flatten(),
    img_to_arr(image20_15test).flatten(),])
# форматируем изображения в бинарные данные
def rgb_to_binary(array):
    for i in range(array.shape[0]):
        if array[i] == 0:
            array[i] = 1
        else:
            array[i] = 0
    return array
# массив для обучения в бинарные
for arr in range(my_array.shape[0]):
    my_array[arr] = rgb_to_binary(my_array[arr])
# массив для теста в бинарные
for arr in range(test_array.shape[0]):
    test_array[arr] = rgb_to_binary(test_array[arr])
# тело нейрона
w = np.zeros(2500).astype(float)
intput = np.array([1, 1, 1])
alfa = 0.1
beta = -5
sigma = lambda x: 1 if x > 0 else 0
def f(x):
    s = beta + np.sum(x @ w)
    return sigma(s)
def train():
    global w
    w_copy = w.copy()
    for x, y in zip(my_array, intput):
        w += alfa * (y - f(x)) * x
    return (w != w_copy).any()
while train():
    pass
print(f(test_array[0]))
print(f(test_array[1]))
print(f(test_array[2]))
print(f(test_array[3]))
print(f(test_array[4]))
print(f(test_array[5]))
print(f(test_array[6]))
# нейрон чувствителен к повороту
