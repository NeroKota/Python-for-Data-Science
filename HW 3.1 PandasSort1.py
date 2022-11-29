import numpy as np
import pandas as pd
books=pd.ExcelFile("knigi.xlsx")
books_prices=books.parse('Worksheet')
D=np.hstack((books_prices.values[:,2:4],books_prices.values[:,4:5])).astype(np.int32)
Y=books_prices.values[:,5:6].astype(np.int32).flatten()
D=D.astype(np.double)
for i in range(len(D[0])):
    a,b = np.polyfit(np.sort(D[:,i]), np.linspace(0,1,len(D[:,i])), 1)
    D[:,i] = D[:,i]*a+b 
X = D.sum(axis=1)
a, b = np.polyfit(X, Y, 1)
for x, y in zip(X, Y):
	print(y, x*a + b)