import numpy as np
import pandas as pd
books=pd.ExcelFile("knigi.xlsx")
books_prices=books.parse('Worksheet')
D=np.hstack((books_prices.values[:,2:4],books_prices.values[:,4:5])).astype(np.double)
Y=books_prices.values[:,5:6].astype(np.int32).flatten()
for i in range(len(D[0])):
    a, b = np.polyfit(
        np.sort(D[:,i]),
        np.linspace(0,1,len(D[:,i])),1
    )
    D[:,i] = D[:,i] * a + b
w = np.zeros((len(D[0]))).astype(np.double)
alfa =  0.2 
betta = -0.4 
sigma = lambda x: x
def f(x):
    s = betta + np.sum(x @ w)
    return sigma(s)
def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y):
        w += alfa * (y - f(x)) * x
    return (w != _w).any()            
while train():
    print(w)
for x, y in zip(D, Y):
    print(x, y, round(f(x)))