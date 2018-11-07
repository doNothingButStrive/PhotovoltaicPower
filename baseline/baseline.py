# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
import datetime

def getDateFeature(data):
    def formatDate(dateStr):
        date_str = dateStr.split('.')
        if len(date_str) == 2:
            d = (datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=1))
        else:
            d = datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S')
        return d
    data['时间'] = data['时间'].apply(lambda x: formatDate(x))
    data['month'] = data['时间'].apply(lambda x: int(x.strftime('%m')))
    data['day'] = data['时间'].apply(lambda x: int(x.strftime('%d')))
    data['hour'] = data['时间'].apply(lambda x: int(x.strftime('%H')))
    data['minute'] = data['时间'].apply(lambda x: int(x.strftime('%M')))
    return data

def baseline():
    for index in range(1, 5):
        dataPath = '../data/train_{}.csv'.format(1)
        
        data = pd.read_csv(dataPath)
        #data = data.drop(['实发辐照度'],axis=1)
        data = getDateFeature(data)
        #data['实发辐照度'] = data['实发辐照度'].apply(lambda x : x if x>=0.0 else np.nan)
        
        #x = np.array(data.drop(['实际功率', '时间'], axis=1))
        x = np.array(data[['month','day','hour','minute','辐照度']])
        #print(x)
        y = np.array(data['实际功率'])
        print(' train begining ')
        
        x_train, x_val = x[:40000], x[40000:]
        y_train, y_val = y[:40000], y[40000:]

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
            
        params = {
                "objective": "regression",
                "metric": "mse",
                "num_leaves": 10,
                "min_child_samples": 100,
                "learning_rate": 0.03,
                "bagging_fraction": 0.8,
                "feature_fraction": 0.8,
                "bagging_frequency": 5,
                "bagging_seed": 666
                }
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_val,
                        early_stopping_rounds=50,
                        verbose_eval=50
                        #categorical_feature=[5,6,7,8]
                        )

        y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        
        plt.plot(y_pred[-60:])
        plt.plot(y_val[-60:])
        plt.show()
        
        print(gbm.feature_importance())
        break

if __name__ == '__main__':
    baseline()





