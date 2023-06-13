import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
class splitData():
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
    
    def split(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=42)

features = ['district_encoder',\
        'area',\
        'new_num_floors',\
        'new_bedrooms',\
        'houseTypes_Bán Luxury home',\
        'houseTypes_Bán Nhà',\
        'houseTypes_Bán Nhà cổ',\
        'houseTypes_Bán Nhà mặt phố',\
        'houseTypes_Bán Nhà riêng']

df = pd.read_excel('HCM_data.xlsx')
df_tranform = pd.DataFrame(data = StandardScaler().fit_transform(df.loc[:, features].values), columns = features)
y = df['price'].values
x = df_tranform[features].values
data = splitData()
data.split(x, y)


y_pred = model.predict(data.X_test)
mse = mean_squared_error(data.y_test, y_pred)
mae = mean_absolute_error(data.y_test, y_pred)
evs = explained_variance_score(data.y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("variance: ", evs)

