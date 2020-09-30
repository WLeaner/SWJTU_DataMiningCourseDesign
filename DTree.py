#定义决策树
class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, data=None):
        #划分值
        self.value = value
        #节点的两个分支
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        #叶子节点的类别
        self.results = results
        #最优划分属性
        self.col = col
        #划分到当前结点的数据
        self.data = data