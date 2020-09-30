from sklearn.model_selection import StratifiedKFold
from load import loadIris,loadWine,loadBreastCancer
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#import scipy.stats as stats

def normal_distribution(a,u,s):
    """
    功能：计算P(a|c),a为样本的某个属性
    a：某个样本的属性a
    u：均值
    s：标准差
    """
    return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-1 * np.power(a - u, 2) / (2 * np.power(s,2)))


def classificationProbability(target):
    """
    功能：获取各类别的概率
    target：标签,ndarray类型
    """
    dit = {}
    length = target.shape[0] #获取到样本总数
    for c in np.unique(target):
        dit[c] = np.sum(target==c) / length
        #print(c,dit[c])
    return dit

def getMeanAndStd(data,target):
    """
    功能：计算各品种的平均值和标准差
    data：数据,ndarray类型
    target：标签,ndarray类型
    """
    dit = {}
    for c in np.unique(target):
        temp = data[target==c]
        dit[c] = np.array([temp.mean(axis=0),temp.std(axis=0)])
        #print(dit[c][0][0],dit[c][1][0])
    return dit

def naiveBayesClassification(test_x,cprobs,mstds):
    """
    功能：朴素贝叶斯分类器
    test：测试集
    cprobs：各类别的概率,字典类型
    mstds：包含各类别各属性各自的平均值和标准差,字典类型
    """
    maxp = 0#最大概率
    dit = {}
    for c,mstd in mstds.items():
        cprob = cprobs[c]
        temp = np.zeros(test_x.shape)
        #计算P(ai|yi)
        for i in range(test_x.shape[1]):
            vfunc = np.frompyfunc(normal_distribution,3,1)
            temp[:,i] = vfunc(test_x[:,i],mstd[0,i],mstd[1,i])
        #计算p(x|yi)p(yi)
        dit[c] = temp.prod(axis=1) * cprob
    nb = pd.DataFrame(dit)
    return nb.columns[np.argmax(nb.values,axis=1)]


def testDataset(data,target):
    """
    功能：对数据集利用朴素贝叶斯进行分类，采用十折交叉验证，且采用F1指标
    data：数据
    target：标签
    """
    skf = StratifiedKFold(n_splits=10 ,random_state=0)
    f1_scores = []
    #10折交叉验证
    for train_index, test_index in skf.split(data,target):
        train_x,test_x = data[train_index], data[test_index]
        train_y, test_y = target[train_index], target[test_index]
        cprobs = classificationProbability(train_y)
        mstds = getMeanAndStd(train_x,train_y)
        pred_y = naiveBayesClassification(test_x,cprobs,mstds)
        #计算f1指标
        f1 = f1_score(test_y, np.array(pred_y), average='macro' )
        f1_scores.append(f1)
    
    return np.array(f1_scores).mean()

if __name__ == "__main__":
    data,target = loadIris()
    f1_mean = testDataset(data,target)
    print('the f1 score of iris is {}'.format(f1_mean))

    data,target = loadWine()
    f1_mean = testDataset(data,target)
    print('the f1 score of wine is {}'.format(f1_mean))

    data,target = loadBreastCancer()
    f1_mean = testDataset(data,target)
    print('the f1 score of breast cancer is {}'.format(f1_mean))