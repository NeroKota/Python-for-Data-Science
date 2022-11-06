import numpy as np
x = np.genfromtxt('D:/Shareman/Programs/Lib/test/Test.csv', delimiter=',', names=True)
np.savetxt('D:/Shareman/Programs/Lib/test/Test.txt', x)
x = np.loadtxt('D:/Shareman/Programs/Lib/test/Test.txt')
lines = len(x)
columns = int(x.size/len(x))
tab = np.eye(lines, columns)
xtab = x*tab
np.savetxt("xtab.csv", xtab, delimiter=",")

