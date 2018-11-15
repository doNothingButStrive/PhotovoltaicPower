# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class LoadData():
    
    def __init__(self, station, dataType='train'):
        self.dataType = dataType
        self.station = station
    
    def load_data(self):
        data = pd.read_csv('../data/{0}_{1}.csv'.format(self.dataType, self.station))
        if self.dataType == 'train':
            data = data[data['辐照度'] != -1.0]
        data = data.reset_index(drop = True)
        return data
        
class FeatureEngineering():
    
    def feature_extract(self, data, dataType='train'):
        if dataType == 'train':
            data = self.drop_duplicate(data)
        data = self.format_date(data)
        data = self.add_poly_features(data, ['风速', '温度', '压强', '湿度'])

        data['dis2peak'] = data['minute'].apply(lambda x: (810 - abs(810 - x)) / 810)
        data = data.drop(['时间'], axis = 1)
        
        return data
    
    def get_hour(self, x):
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
    
    def format_date(self, data):
        data['year'] = data['时间'].apply(lambda x: x[0:4]).astype('int32')
        data['month'] = data['时间'].apply(lambda x: x[5:7]).astype('int32')
        data['day'] = data['时间'].apply(lambda x: x[8:10]).astype('int32')
        data['minute'] = data['时间'].apply(lambda x: self.get_hour(x)).astype('int32')
        return data
    
    def add_poly_features(self, data, column_names):
        features = data[column_names]
        rest_features = data.drop(column_names, axis=1)
        poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                     columns=poly_transformer.get_feature_names(column_names))
        
        for col in poly_features.columns:
            rest_features.insert(1, col, poly_features[col])
        return rest_features
    
    def drop_duplicate(self, data):
        pre = -100.0
        count = 0
        index_list = []
        tmp_list = []
        for index in data.index:
            fzd = data.loc[index][-1]
            if pre != fzd and pre != -100:
                if count >= 12:
                    index_list.extend(tmp_list)
                    del tmp_list[:]
                count = 0
                del tmp_list[:]
            pre = fzd
            count += 1
            tmp_list.append(index)
        
        data = data.drop(index_list)
        return data
    
if __name__ == '__main__':
    #dataType = 'test'
    station = 1
    ld = LoadData(station)
    fn = FeatureEngineering()
    data = ld.load_data()
    print(data.shape)
    data = fn.feature_extract(data)
    print(data.head(5))



