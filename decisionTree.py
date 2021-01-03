from load import loadIris,loadWine,loadBreastCancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from DTree import Tree
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculateDiffCount(datas):
    """
    功能：各品种的个数
    返回：字典{type1:type1Count,type2:type2Count ... typeN:typeNCount}
    """
    results = {}
    for data in datas:
        # data[-1] means dataType
        if data[-1] not in results:
            results[data[-1]] = 1
        else:
            results[data[-1]] += 1
    return results


def gini(rows):
    """
    功能：计算基尼指数
    """
    length = len(rows)
    results = calculateDiffCount(rows)
    imp = 0.0
    for i in results:
        imp += (results[i] / length)**2
    return 1 - imp


def splitDatas(rows, value, column):
    """
    功能：连续变量离散化(二分)
    """
    return (rows[rows[:,column] >= value],rows[rows[:,column] < value])


def treeGenerate(rows, evaluationFunction=gini):
    """
    功能：建立决策树
    """
    #计算当前的基尼指数
    currentGain = evaluationFunction(rows)
    rows_length,column_length = rows.shape[0],rows.shape[1]
    best_gain,best_value,best_set = 0.0,None,None

    for col in range(column_length - 1):
        col_value_set = np.unique(rows[:,col])
        #寻找某个属性的最佳划分点
        for value in col_value_set:
            list1, list2 = splitDatas(rows, value, col)
            #计算基尼指数          
            p = len(list1) / rows_length
            gain = currentGain - p * evaluationFunction(list1) - (1 - p) * evaluationFunction(list2)
            if gain > best_gain:#gain越大，基尼指数越小
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)

    if best_gain > 0:
        trueBranch = treeGenerate(best_set[0], evaluationFunction)
        falseBranch = treeGenerate(best_set[1], evaluationFunction)
        return Tree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch)
    else:
        return Tree(results=calculateDiffCount(rows), data=rows)


def prune(tree, miniGain, evaluationFunction=gini):
    """
    功能：后剪枝
    """
    #还未到达叶子结点
    if tree.trueBranch.results == None: prune(tree.trueBranch, miniGain, evaluationFunction)
    if tree.falseBranch.results == None: prune(tree.falseBranch, miniGain, evaluationFunction)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = tree.trueBranch.data.shape[0]
        len2 = tree.falseBranch.data.shape[0]
        len3 = np.vstack((tree.trueBranch.data,tree.falseBranch.data)).shape[0]
        p = float(len1) / (len1 + len2)
        gain = evaluationFunction(np.vstack((tree.trueBranch.data,tree.falseBranch.data))) - p * evaluationFunction(
            tree.trueBranch.data) - (1 - p) * evaluationFunction(tree.falseBranch.data)
        if (gain < miniGain):
            tree.data = np.vstack((tree.trueBranch.data,tree.falseBranch.data))
            #剪枝后用子树中大多数训练样本所属的类别来标识
            temp = [[key,value] for key,value in calculateDiffCount(tree.data).items()]
            temp.sort(key = lambda x:x[1],reverse = True)
            tree.results = {temp[0][0]:temp[0][1]}
            tree.trueBranch = None
            tree.falseBranch = None


def classify(data, tree):
    """
    功能：对数据进行分类
    """
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if v >= tree.value:
            branch = tree.trueBranch
        else:
            branch = tree.falseBranch
        return classify(data, branch)


def testDataset(data,target):
    """
    功能：对数据集利用朴素贝叶斯进行分类，采用十折交叉验证，且采用F1指标
    data：数据
    target：标签
    cut：决定是否剪枝，默认为False即不剪枝
    """
    skf = StratifiedKFold(n_splits=10 ,random_state=0,shuffle=True)
    f1_scores = []
    #10折交叉验证
    for train_index, test_index in skf.split(data,target):
        train_x,test_x = data[train_index], data[test_index]
        train_y, test_y = target[train_index], target[test_index]
        train_y = train_y.reshape(train_y.shape[0],-1)
        trainingData = np.hstack((train_x,train_y))
        decisionTree = treeGenerate(trainingData, evaluationFunction=gini)
        prune(decisionTree, 0.25)
        pred_y = []
        for row in test_x:
            max_val = 0
            pred_y += [int(key) for key in classify(row,decisionTree).keys()]
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