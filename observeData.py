# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

columns = ['辐照度', '风速', '风向', '温度', '压强', '湿度', '实发辐照度' ,'实际功率']

def observeData(station, col):
    data = pd.read_csv('data/train_{}.csv'.format(station))
    feature = data[data['辐照度'] != -1.0][col]
    
    '''
    def wind(x):
        return (x) // 45
    if col == '风向':
        feature = feature.apply(lambda x : wind(x))
    '''
    d = 7 * 24 * 4
    for index in range(0, len(feature)-d, d):
        tmp = feature[index:index+d]
        plt.figure(1, (10,10))
        plt.plot(tmp)
        plt.show()
        #time.sleep(5)
        print()
        break
    
    print(' ========{0}-{1}========'.format(col, 'all'))
    plt.figure(1,(10, 10))
    plt.hist(feature, bins=20)
    #plt.scatter(range(len(feature)), feature, s=3)
    plt.show()
    
    
def pearson(dataType):
    data = pd.read_csv('data/train_{}.csv'.format(dataType))
    data = data.drop(['时间'], axis=1)
    corr = data.corr()
    print(corr[['实发辐照度','实际功率']])
    #print(corr)

if __name__ == '__main__':
    index = 1
    for station in range(1, 5):
        observeData(4, columns[index])
        break
    #pearson(1)
    '''
    data = pd.read_csv('data/train_{}.csv'.format(1))
    #res = data[data['辐照度'] == -1.0]['实际功率']
    res = data[data['辐照度'] == -1.0]['实发辐照度']
    plt.scatter(range(len(res)), res)
    plt.show()
    '''
    