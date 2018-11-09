# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import PolynomialFeatures

class LoadData():
    
    def __init__(self, dataType):
        self.dataType = dataType
    
    def loadData(self, station):
        pass
    
    def loadAllData(self):
        data1 = pd.read_csv('../data/{}_1.csv'.format(self.dataType))
        data2 = pd.read_csv('../data/{}_2.csv'.format(self.dataType))
        data3 = pd.read_csv('../data/{}_3.csv'.format(self.dataType))
        data4 = pd.read_csv('../data/{}_4.csv'.format(self.dataType))
        if self.dataType == 'train':
            data3 = data3[]
        

def get_hour(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return h * 60 + m

def format_date(data):
    data['year'] = data['时间'].apply(lambda x: x[0:4]).astype('int32')
    data['month'] = data['时间'].apply(lambda x: x[5:7]).astype('int32')
    data['day'] = data['时间'].apply(lambda x: x[8:10]).astype('int32')
    data['minute'] = data['时间'].apply(lambda x: get_hour(x)).astype('int32')
    return data
    '''
    def formatTime(dateStr):
        date_str = dateStr.split('.')
        if len(date_str) == 2:
            d = (datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=1))
        else:
            d = datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S')
        return d
    data['时间'] = data['时间'].apply(lambda x: formatTime(x))
    data['year'] = data['时间'].apply(lambda x: int(x.strftime('%Y')))
    data['month'] = data['时间'].apply(lambda x: int(x.strftime('%m')))
    data['day'] = data['时间'].apply(lambda x: int(x.strftime('%d')))
    #data['hour'] = data['时间'].apply(lambda x: int(x.strftime('%H')))
    data['minute'] = data['时间'].apply(lambda x: int(x.strftime('%M')) + 60*int(x.strftime('%H')) )
    return data
    '''
    
def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features

def getFeature(station):
    trainDataPath = '../data/train_{}.csv'.format(station)
    testDataPath = '../data/test_{}.csv'.format(station)
    
    train_data = pd.read_csv(trainDataPath)
    test_data = pd.read_csv(testDataPath)

    # format date
    train_data = format_date(train_data)
    test_data = format_date(test_data)
    train_data['dis2peak'] = train_data['minute'].apply(lambda x: (810 - abs(810 - x)) / 810)
    test_data['dis2peak'] = test_data['minute'].apply(lambda x: (810 - abs(810 - x)) / 810)
  
    
    # polynomial feature
    train_data = add_poly_features(train_data, ['风速', '风向','温度', '压强', '湿度'])
    test_data = add_poly_features(test_data, ['风速', '风向','温度', '压强', '湿度'])
    '''
    train_data = add_poly_features(train_data, ['风速', '风向'])
    train_data = add_poly_features(train_data, ['温度', '压强', '湿度'])
    test_data = add_poly_features(test_data, ['风速', '风向'])
    test_data = add_poly_features(test_data, ['温度', '压强', '湿度'])
    '''
    train_data, val_data = train_data[train_data.year < 2018], train_data[train_data.year == 2018]
    
    train_data = train_data.drop(['时间', '实发辐照度'], axis=1)
    val_data = val_data.drop(['时间', '实发辐照度'], axis=1)
    
    available_train_data = train_data[train_data['辐照度'] != -1.0]
    #available_train_data = train_data
    analysis_train_data = train_data[train_data['辐照度'] == -1.0]
    #available_val_data = val_data
    available_val_data = val_data[val_data['辐照度'] != -1.0]
    analysis_val_data = val_data[val_data['辐照度'] == -1.0]
   
    featurePath = 'data/{}'.format(station)
    if not os.path.exists(featurePath):
        os.makedirs(featurePath)
    available_train_data.to_csv('data/{}/available_train_data.csv'.format(station))
    analysis_train_data.to_csv('data/{}/analysis_train_data.csv'.format(station))
    available_val_data.to_csv('data/{}/available_val_data.csv'.format(station))
    analysis_val_data.to_csv('data/{}/analysis_val_data.csv'.format(station))
    test_data.to_csv('data/{}/test_data.csv'.format(station))
    
if __name__ == '__main__':
    station = 1
    getFeature(1)











