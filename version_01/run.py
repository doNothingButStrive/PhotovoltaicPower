# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

def loadData(station):
    available_train_data = pd.read_csv('data/{}/available_train_data.csv'.format(station), index_col=0)
    analysis_train_data = pd.read_csv('data/{}/analysis_train_data.csv'.format(station), index_col=0)
    available_val_data = pd.read_csv('data/{}/available_val_data.csv'.format(station), index_col=0)
    analysis_val_data = pd.read_csv('data/{}/analysis_val_data.csv'.format(station), index_col=0)
    return available_train_data,analysis_train_data,available_val_data,analysis_val_data

class LGBModel():
    
    params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 30,
            "min_child_samples": 100,
            "learning_rate": 0.01,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "bagging_frequency": 5,
            "bagging_seed": 666,
            "verbosity": -1
            }
    
    def __init__(self):
        pass
    
    def train(self, x_train, y_train, x_val, y_val):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
        
        print('begin train')
        gbm = lgb.train(self.params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100
                    )
        
        y_pred = gbm.predict(x_val)
        plt.plot(y_pred[-60:])
        plt.plot(y_val[-60:])
        plt.show()
        print(gbm.feature_importance())
        
    def predict(self):
        pass
        
if __name__ == '__main__':
    station = 1
    available_train_data,analysis_train_data,available_val_data,analysis_val_data = loadData(station)

    x_train, y_train = available_train_data.drop(['实际功率','year'], axis=1), available_train_data['实际功率']
    x_val, y_val = available_val_data.drop(['实际功率','year'], axis=1), available_val_data['实际功率']

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)

    lgbmodel = LGBModel()
    lgbmodel.train(x_train, y_train, x_val, y_val)

    