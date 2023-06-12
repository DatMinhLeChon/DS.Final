
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
df_tranform = pd.DataFrame(data = StandardScaler().fit_transform(df.loc[:, features].values), columns = features)

class splitData():
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.y_test = []
        self.x_test = []
        
    def split(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=42)

class LinearModel():
    def __init__(self):
        self._X = []
        self._y = []
        self.model = None
    
    def linear(self):
        reg = LinearRegression().fit(self._X, self._y)
        self.model = reg
    
    def predict(self, x_test, y_test):
        df_temp = pd.DataFrame(columns = ['predict', 'real', 'error'])
        for x, y in zip(x_test, y_test):
            df_temp= pd.concat([df_temp, pd.DataFrame(data=[[self.model.predict(np.array([x])), y, abs(self.model.predict(np.array([x]))- y) ]], columns = ['predict', 'real', 'error'] )], axis =0)
        return df_temp
        
if __name__ =="__main__":
    linear_model = LinearModel()
    y = df['price'].values
    x = df_tranform[features].values
    data = splitData()
    data.split(x, y)
    linear_model._X = data.x_train
    linear_model._y = data.y_train
    linear_model.linear()
    df_predict = linear_model.predict(data.x_test, data.y_test)
    mse, bias, var = bias_variance_decomp(linear_model.model, data.x_train, data.y_train, data.x_test, data.y_test, loss='mse', num_rounds=200, random_seed=1)
    print("MSE:", mse , '\n','Bias:', bias, '\n', 'variance:', var)
    
    df_predict.hist()
    
    
# python model/linear/linear.py