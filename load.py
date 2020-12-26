import pandas as pd
import numpy as np

def standardization(data):
    """
    z-score归一化,即（X-Mean）/(Standard deviation)
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def loadIris():
    """
    功能：加载鸢尾花卉数据集的数据和标签
    """
    iris = pd.read_csv('./data/iris.csv')
    x,y = np.array(iris.iloc[:,0:4]),np.array(iris.iloc[:,[4]]).flatten()
    x = standardization(x)
    return x,y

def loadWine():
    """
    功能：加载葡萄酒数据集的数据和标签
    """
    wine = pd.read_csv('./data/wine.csv')
    x,y = np.array(wine.iloc[:,1:]),np.array(wine.iloc[:,[0]]).flatten()
    x = standardization(x)
    return x,y


def loadBreastCancer():
    """
    功能：加载威斯康星乳腺癌数据集的数据和标签
    """
    breast_cancer = pd.read_csv('./data/breast-cancer-wisconsin.csv')
    #删除含有缺失值的行
    breast_cancer.replace('?',np.NaN,inplace=True)
    breast_cancer.dropna(inplace=True)
    x,y = np.array(breast_cancer.iloc[:,1:10],dtype='int64'),np.array(breast_cancer.iloc[:,[10]]).flatten()
    x = standardization(x)
    return x,y

if __name__ == "__main__":
    data,target = loadIris()
    print(data.dtype)