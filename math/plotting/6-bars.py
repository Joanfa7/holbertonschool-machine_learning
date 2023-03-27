#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

lables = ['Farah', 'Fred', 'Felicia']
width = 0.5

apples = fruit[0][:]
bananas = fruit[1][:]
orange = fruit[2][:]
peaches = fruit[3][:]

plt.ylabel('Quantity of Fuits')
plt.title('Number of Fuit per Person')
plt.bar(lables, apples, width, color='red')
plt.bar(lables, bananas, width, bottom=apples, color='yellow')
plt.bar(lables, orange, width, bottom=apples + bananas, color='#ff8000')
plt.bar(
    lables,
    peaches,
    width,
    bottom=apples +
    bananas +
    orange,
    color='#ffe5b4')

plt.yticks(np.arange(0, 90, step=10))
plt.legend(['apples', 'bananas', 'oranges', 'peaches'])

plt.show()
