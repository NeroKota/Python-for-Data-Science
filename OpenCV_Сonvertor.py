# импорт необходимых библиотек
import glob as gb
import cv2 as cv
import os
from pathlib import Path
# создание списка с названиями зображений
list = []
file_name = []
JPG = gb.glob('D:/Shareman/Programs/Lib/test/Images_JPG/*.jpg')  
for image in JPG:
    list.append(cv.imread(image))  
    file_name.append(Path(image).stem) 
# сохранение изображений нового формата 
dir = "D:/Shareman/Programs/Lib/test/Images_PNG"
if not os.path.isdir(dir):
    os.mkdir(dir)
for i in range(len(JPG)):
    cv.imwrite(f'{dir}/{file_name[i]}.png', list[i])