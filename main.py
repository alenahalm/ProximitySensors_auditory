import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

def func(x, a, b):
    return a / (x + b)

x = []
y = []

with open('a.txt') as f:
    for line in f.readlines():
        a, b = line.strip().split(', ')
        x.append(float(a))
        y.append(float(b))

# M = np.vstack([x, np.ones(len(x))]).T

# alpha = np.linalg.lstsq(M, np.cosh(y), rcond=None)[0]

# print(alpha)
x = np.array(x)
y = np.array(y)

plt.plot(x, y, 'b-', label='data')

popt, pcov = curve_fit(func, x, y)
print(popt)

for i in range(len(y)):
    y[i] = func(x[i], popt[0], popt[1])

plt.plot(x, y)

# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
plt.show()



# plt.scatter(x, y, color='red')
# plt.scatter(x, )
# plt.show()
