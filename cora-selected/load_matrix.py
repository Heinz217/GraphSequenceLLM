import torch

laplacian = torch.load('C:/Users/86130/PycharmProjects/pythonProject30/cora-selected/laplacian_2_10.pt')
laplacian1 = torch.load('C:/Users/86130/PycharmProjects/pythonProject30/cora-selected/laplacian_2_5.pt')
laplacian2 = torch.load('C:/Users/86130/PycharmProjects/pythonProject30/cora-selected/laplacian_2_20.pt')

print(laplacian)
print(laplacian[0][0:110])

for i in range(0, 110):
    print(laplacian[i][0:110])


# using matplotlib to plot the laplacian matrix
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(laplacian, cmap='hot', interpolation='nearest')
plt.show()

# changing the color map to 'cool'
plt.imshow(laplacian, cmap='cool', interpolation='nearest')
plt.show()