import numpy as np

input = np.genfromtxt('gamingData5.csv', skip_header=1, delimiter=';')
for i, x in enumerate(np.isnan(input)):
    print(f"{input[i]}")