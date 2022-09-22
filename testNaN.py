import numpy as np
import pandas as pd 

# search for NULL values using numpy
input = np.genfromtxt('gamingData5.csv', skip_header=1, delimiter=';')


# seraching for NULL values and remove them using pandas
data = pd.read_csv('gamingData5.csv', delimiter=';')
data.dropna(how='any', axis=0, inplace=True)
print(data['Q8'])