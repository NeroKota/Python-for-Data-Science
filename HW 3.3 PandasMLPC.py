from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
books=pd.ExcelFile("knigi.xlsx")
books_prices=books.parse('Worksheet')
D=np.hstack((books_prices.values[:,2:4],books_prices.values[:,4:5])).astype(float)
Y0=books_prices.values[:,5:6].astype(np.int).flatten()
HP = sorted(set(Y0)) # Год издания книги
for i in range(len(D[0])):
    a = np.polyfit(np.sort(D[:,i]),np.linspace(0,1,len(D[:,i])),1)
    D[:,i] = D[:,i] * a[0] + a[1]
clasification = MLPClassifier(
    solver='lbfgs', 
    hidden_layer_sizes=(200), 
    random_state=1,
    max_iter=100, 
    warm_start=True
)
clasification.fit(D, Y0)
Year = clasification.predict(D)
for i in range(len(Y0)):
 print(f'Год издания книги = {Y0[i]}, Год, который получился = {Year[i]}')
 #на удивление, даже достаточно точно
