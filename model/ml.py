import keras 
import torch
import pandas as pd
from torch import nn 
import numpy as np
from sklearn.linear_model import LinearRegression

features = ['price',\
        'city',\
        'city_target_encoder',\
        'city_index',\
        'area',\
        'new_num_floors',\
        'new_bedrooms',\
        'houseTypes_Bán Luxury home',\
        'houseTypes_Bán Nhà',\
        'houseTypes_Bán Nhà cổ',\
        'houseTypes_Bán Nhà mặt phố',\
        'houseTypes_Bán Nhà riêng']

df = pd.read_excel('final_data.xlsx')
class ReSystem():
    def __init__(self):
        
        return 

class LinearModel():
    def __init__(self):
        self._X = []
        self._y = []
    
    def linear(self):
        reg = LinearRegression().fit(self._X, self._y)
        return reg
if __name__ =="__main__":
    model = LinearModel()
    model._X = df['price']
    model._Y = 