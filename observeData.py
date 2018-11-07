# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def observeData(dataType):
    data = pd.read_csv('data/train_{}.csv'.format(dataType))
    columns = data.columns[1:]
    for index in columns:
        feature = data[index]
        feature = feature.iloc[:30*24*4]
        
        print(' ========{}========'.format(index))
        plt.figure(1,(10, 10))
        #plt.scatter(range(len(feature)), feature, s=3)
        plt.plot(feature)
        plt.show()
        print()

def pearson(dataType):
    data = pd.read_csv('data/train_{}.csv'.format(dataType))
    data = data.drop(['时间'], axis=1)
    corr = data.corr()
    #print(corr[['实发辐照度','实际功率']])
    print(corr)

if __name__ == '__main__':
    dataType = 1
    observeData(dataType)
    pearson(1)
    