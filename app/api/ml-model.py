import numpy as np
from numpy.core.fromnumeric import size
# import sklearn
import pandas as pd

data = pd.read_csv('EuroMillions_numbers.csv',delimiter=";")
print(data)

size_of_dataset = data.shape[1]

def create_loosing_number(size_of_dataset : int):
    loosing_data = []
    for i in range(size):
        loosing_data[i]= [np.random.uniform(low=1,high=50),np.random.uniform(low=1,high=50),np.random.uniform(low=1,high=50),np.random.uniform(low=1,high=50),np.random.uniform(low=1,high=50),np.random.uniform(low=1,high=12),np.random.uniform(low=1,high=12)]
    return loosing_data

class Model(BaseModel):
    Metrics: List[float]
    name : str
    trainMetrics : List[float]