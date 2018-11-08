# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['辐照度', '风速', '风向', '温度', '压强', '湿度', '实发辐照度' ,'实际功率']

def observeData(station, col):
    data = pd.read_csv('data/train_{}.csv'.format(station))
    
    feature = np.array(data[data['辐照度'] != -1.0][col])

    #feature = np.log(feature+2)
    tmp = feature[:7*24*4]
    print(' ========{0}-{1}========'.format(col, 'all'))
    plt.figure(1,(10, 10))
    plt.hist(feature, bins=20)
    plt.show()
    
    print(' ========{0}-{1}========'.format(col, 'tmp'))
    plt.figure(1,(10, 10))
    plt.scatter(range(len(tmp)), tmp, s=3)
    plt.plot(tmp)
    plt.show()

def pearson(dataType):
    data = pd.read_csv('data/train_{}.csv'.format(dataType))
    data = data.drop(['时间'], axis=1)
    corr = data.corr()
    #print(corr[['实发辐照度','实际功率']])
    print(corr)

if __name__ == '__main__':
    
    index = 0
    station = 2
    observeData(station, columns[index])
    #pearson(1)
    '''
    data = pd.read_csv('data/train_{}.csv'.format(1))
    #res = data[data['辐照度'] == -1.0]['实际功率']
    res = data[data['辐照度'] == -1.0]['实发辐照度']
    plt.scatter(range(len(res)), res)
    plt.show()
    '''
    