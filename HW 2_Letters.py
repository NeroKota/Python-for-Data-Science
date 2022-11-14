import numpy as np

weight0 = np.zeros((16)) #Вес0
weight1 = np.zeros((16)) #Вес1

Letters = np.array([
   [0,1,1,1,
    1,0,0,0,
    1,0,1,1,
    0,1,1,1,], #G1
   [1,1,1,1,
    1,0,0,0,
    1,0,1,1,
    1,1,1,1,], #G2
   [1,1,1,1,
    1,0,0,0,
    1,0,0,0,
    1,1,1,1,], #C1
   [0,1,1,1,
    1,0,0,0,
    1,0,0,0,
    0,1,1,1,], #C2
]) #Входной вектор (буквы C, G)

y0 = np.array([1,1,0,0]) #выходной вектор 0 (G)
y1 = np.array([0,0,1,1]) #выходной вектор 1 (C)

alfa =  0.2 #скорость обучения
beta = -0.4 #коэффициент торможения
sigma = lambda x: 1 if x > 0 else 0 #функция активации

def f0(x):
    s = beta + np.sum(x @ weight0)
    return sigma(s) #тело 0 

def f1(x):
    s = beta + np.sum(x @ weight1) 
    return sigma(s) #тело 1 

def train0():
    global weight0
    _weight = weight0.copy()
    for x, y in zip(Letters, y0):
        weight0 += alfa * (y - f0(x)) * x
    return (weight0 != _weight).any() #эпоха обучения 0 

def train1():
    global weight1
    _weight = weight1.copy()
    for x, y in zip(Letters, y1):
        weight1 += alfa * (y - f1(x)) * x
    return (weight1 != _weight).any() #эпоха обучения 1 

while train0() and train1():
    print(weight0, weight1) #перебор

Letters_test = np.array([
   [0,1,1,1,
    1,0,0,1,
    1,0,0,1,
    0,1,1,1,],
   [0,1,1,1,
    1,0,0,0,
    1,0,0,0,
    0,1,1,1,],
   [0,1,1,1,
    1,0,0,0,
    1,0,1,1,
    0,1,1,1,],
   [0,1,1,1,
    1,0,0,0,
    1,0,1,1,
    0,1,1,1,],])

for x in Letters_test:
    print(x, f0(x), f1(x)) #соответствие/несоответствие
    
