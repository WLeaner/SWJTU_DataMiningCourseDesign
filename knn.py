import pandas as pd
import numpy as np
from load import loadIris,loadWine,loadBreastCancer
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


def distance(train,test):
    """
    功能：计算欧式距离
    test：测试数据
    train：训练集
    """
    return cdist(train,test,metric='euclidean')

def knnClassification(train_x,train_y,test_x,test_y,k):
    """
    功能：KNN分类
    train：训练集
    test：测试集
    k：算法的参数k
    """
    edis = distance(train_x,test_x)
    pred_y = []
    for i in range(edis.shape[1]):
        df = pd.DataFrame({'dis':edis[:,i],'label':train_y})
        #按距离从小到大进行排序
        df.sort_values(by='dis',inplace=True)
        #取出离样本最近的k个点并进行分组
        groups = df.iloc[:k,:].groupby('label').count()
        #取出最大的分组的标签，即预测值
        pred = groups.sort_values('dis',ascending=False).index[0]
        pred_y.append(pred)

    return f1_score(test_y, np.array(pred_y), average='macro' )

def testDataset(data,target,k):
    """
    功能：对数据集利用KNN进行分类，采用十折交叉验证，且采用F1指标
    func：加载属性函数
    k：多少折，默认为10
    """
    skf = StratifiedKFold(n_splits=10 ,random_state=0)
    f1_scores = []
    #10折交叉验证
    for train_index, test_index in skf.split(data,target):
        train_x,test_x = data[train_index], data[test_index]
        train_y, test_y = target[train_index], target[test_index]
        f1 = knnClassification(train_x,train_y,test_x,test_y,k)
        f1_scores.append(f1)
    
    return np.array(f1_scores).mean()

if __name__ == "__main__":
    #尝试k取值从1到19
    data,target = loadIris()
    for i in range(1,20):
        f1_mean = testDataset(data,target,i)
        print('k = {},  the mean f1 score of iris is {}'.format(i,f1_mean))

    data,target = loadWine()
    for i in range(1,20):
        f1_mean = testDataset(data,target,i)
        print('k = {}, the mean f1 score of wine is {}'.format(i, f1_mean))

    data,target = loadBreastCancer()
    for i in range(1,20):
        f1_mean = testDataset(data,target,i)
        print('k = {}, the mean f1 score of breast cancer is {}'.format(i,f1_mean))
