import keras 
import torch
import pandas as pd
from torch import nn 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features = ['city_target_encoder',\
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

class splitData():
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.y_test = []
        self.x_test = []
        
    def split(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
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
    
    x = df['price'].values
    y = df[features].values
    data = splitData()
    data.split(x, y)
    model._X = data.x_train
    model._y = data.y_train
    
    print(model.linear())