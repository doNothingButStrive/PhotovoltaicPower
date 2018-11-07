# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
import datetime

def formatDate(dateStr):
    date_str = dateStr.split('.')
    if len(date_str) == 2:
        d = (datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=1))
    else:
        d = datetime.datetime.strptime(date_str[0], '%Y-%m-%d %H:%M:%S')
    return d
    

def getFeature():
    dataPath = '../data/train_{}.csv'.format(1)
    data = pd.read_csv(dataPath)
    
    data['时间'] = data['时间'].apply(lambda x: formatDate(x))
    data['year'] = data['时间'].apply(lambda x: int(x.strftime('%Y')))
    data['month'] = data['时间'].apply(lambda x: int(x.strftime('%m')))
    data['day'] = data['时间'].apply(lambda x: int(x.strftime('%d')))
    data['hour'] = data['时间'].apply(lambda x: int(x.strftime('%H')))
    data['minute'] = data['时间'].apply(lambda x: int(x.strftime('%M')))
    print(data)
    
    return
    data['实发辐照度'] = data['实发辐照度'].apply(lambda x : x if x>=0.0 else np.nan)
        
    x = np.array(data.drop(['实际功率','实发辐照度', '时间'], axis=1))
    #x = np.array(data.drop(['实际功率', '时间'], axis=1))

    print(x.shape)
    y = np.array(data['实际功率'])
    print(' train begining ')

if __name__ == '__main__':
    getFeature()











