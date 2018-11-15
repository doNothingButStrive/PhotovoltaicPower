# -*- coding: utf-8 -*-

import preprocess
import model
import os
import lightgbm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train():
    for station in range(1, 5):
        ld = preprocess.LoadData(station)
        fe = preprocess.FeatureEngineering()
        lgb = model.LGBModel()
        
        data = ld.load_data()
        #print(data.shape)
        data = fe.feature_extract(data)
        #print(data.shape)
        
        x = data.drop(['实际功率', '实发辐照度'], axis = 1)
        y = data['实际功率']
        y = data['实发辐照度']
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=678)
        #print(x_train.shape, x_val.shape)
        
        #print(x_train.head(5))
        gbm = lgb.train(x_train, y_train, x_val ,y_val)
        '''
        dumpPath = 'model/'
        if not  os.path.exists(dumpPath):
            os.makedirs(dumpPath)
        with open(dumpPath+'lgb_{}.model'.format(station), 'wb') as f:
            pickle.dump(gbm, f)
        #gbm.save_model(, num_iteration=gbm.best_iteration)
        '''

def predict():
    for station in range(1 ,5):
        ld = preprocess.LoadData(station, 'test')
        fe = preprocess.FeatureEngineering()
        with open('model/lgb_{}.model'.format(station), 'rb') as f:
            gbm = pickle.load(f)
        #gbm = lightgbm.Booster(model_file=)
        data = ld.load_data()
        data = fe.feature_extract(data, 'test')
        
        test_id = data['id']
        del_id = data[data['辐照度'].isin([-1.0])]['id']
        test = data.drop(['id'], axis=1)
        
        republish_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
        republish_pred = pd.DataFrame(republish_pred)
        sub = pd.concat([test_id, republish_pred], axis=1)
        print(sub.shape)
        sub.columns = ['id', 'predicition']
        sub.loc[sub['id'].isin(del_id), 'predicition'] = 0.0
        sub.to_csv('submit/version_{}.csv'.format(station), index=False, sep=',', encoding='UTF-8') 


if __name__ == '__main__':
    train()
    '''
    predict()
    data1 = pd.read_csv('submit/version_1.csv')
    data2 = pd.read_csv('submit/version_2.csv')
    data3 = pd.read_csv('submit/version_3.csv')
    data4 = pd.read_csv('submit/version_4.csv')
    res = pd.concat([data1, data2, data3, data4], axis = 0)
    res.to_csv('submit/res.csv', index=False, encoding='utf-8')
    '''



