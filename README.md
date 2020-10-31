# SWJTU_DataMiningCourseDesign

声明：本项目只是单纯的编程实践，为了帮助理解也写了下面的说明文档，其中有不少内容出自其他博客(见文末参考博客)
若公式无法显示，建议下载插件Maxjax Plugin for Github

环境介绍：Python3.6.3

涉及的库：pandas, sklearn, numpy .etc

# 一.数据集介绍

本文中使用的数据集breast cancer, iris和wine都是来自于UCI，下面是对这三个数据集的详细介绍：

### 1.1 Breast Cancr数据集

威斯康星乳腺癌数据集一共包含569个恶性或者良性肿瘤细胞样本，分为两类，总共包含6个属性，该数据集的部分数据展示如下：

![breast_cancer](https://i.loli.net/2020/09/26/1NaUlwb6FpoiEHc.png)

其中各个属性的描述具体见下表：

| 属性    | 属性描述                                      |
| ------- | --------------------------------------------- |
| C_D     | Sample code number，样本代码编号              |
| C_T     | Clump Thickness，肿块厚度                     |
| U_C_Si  | Uniformity of Cell Size，细胞大小的均匀性     |
| U_C_Sh  | Uniformity of Cell Shape，细胞形状的均匀性    |
| M_A     | Marginal Adhesion，边缘粘                     |
| S_E_C_S | Single Epithelial Cell Size，单个上皮细胞大小 |
| B_N     | Bare Nuclei，裸核                             |
| B_C     | Bland Chromatin，乏味染色体                   |
| N_N     | Normal Nucleoli，正常核                       |
| M       | Mitoses，有丝分裂                             |
| Class   | 类别(2代表良性，4代表恶性)                    |

### 1.2 Iris数据集

鸢尾花卉数据集一共包含150个样本，分为三类(Setosa，Versicolour，Virginica)，每类50个数据，该数据集的部分数据展示如下：

![iris](https://i.loli.net/2020/09/26/Y6o1CKg7VLOUeqm.png)

其中各个属性的描述具体见下表：

| 属性         | 属性描述                                             |
| ------------ | ---------------------------------------------------- |
| sepal.length | 花萼长度                                             |
| sepal.width  | 花萼宽度                                             |
| petal.length | 花瓣长度                                             |
| petal.width  | 花瓣宽度                                             |
| variety      | 花类型(0表示Setosa, 1表示Versicolour 2表示Virginica) |

### 1.3 Wine数据集

葡萄酒数据集包含178个样本，也分为三类(1, 2, 3)，其中第一类包含59个样本，第二类包含71个样本，第三类包含48个样本，在该数据集中包含了三种酒13种不同成分的数量，下面是该数据集的部分数据：

![wine](https://i.loli.net/2020/09/26/4YIQL2PvXkhTgCz.png)

其中各属性的描述如下：

| 属性                | 属性描述                |
| ------------------- | ----------------------- |
| Wine                | 类别                    |
| Alcohol             | 酒精                    |
| Malic.acid          | 苹果酸                  |
| Ash                 | 灰                      |
| Acl                 | 灰分的碱度              |
| Mg                  | 镁                      |
| Phenols             | 总酚                    |
| Flavanoids          | 黄酮类化合物            |
| Noflavanoid.phenols | 非黄烷类酚类            |
| Proanth             | 原花色素                |
| Color.int           | 颜色强度                |
| Hue                 | 色调                    |
| OD                  | 稀释葡萄酒的OD280/OD315 |
| Proline             | 脯氨酸                  |



# 二.算法介绍

## 2.1 KNN算法

KNN，即K-Nearest Neighbors，其核心思路为：一个样本与数据集中的K个样本最相似，如果这K个样本中的大多数属于某个类别，则该样本也属于这个类别。如下图所示，绿色圆要被决定赋予哪个类，当K=3时，由于红色三角形占大多数，因此绿色圆属于红色三角形，如果K=5，由于蓝色正方形占大多数，因此绿色圆属于蓝色正方形类。

![img](https://i.loli.net/2020/09/26/Elg8y6psaWHf4V2.jpg)

由此说明KNN算法的结果很大程度上取决于K的选择，而在KNN算法中度量对象间的相似性一般采用欧式距离或曼哈顿距离。在训练集中的数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：

1. 计算测试数据与各个训练数据之间的距离；
2. 按照距离的递增关系进行排序；
3. 选取距离最小的K个点；
4. 确定前K个点所在类别的出现频率；
5. 返回前K个点中出现频率最高的类别作为测试数据的预测分类。

## 2.2 朴素贝叶斯

朴素贝叶斯的思想：对于给出的待分类选项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。其正式定义为：

1. 设一个待分类项$x=\lbrace a1,a2,...,am \rbrace$，而每个$aj$为$x$的一个特征属性
2. 有类别集合$c=\lbrace y1,y2,...,yn \rbrace$
3. 计算$P(y1|x),P(y2|x),...,P(yn|x)$
4. 如果$P(yi|x)=max\lbrace P(y1|x),P(y2|x),...,P(yn|x)\rbrace$，则$x$属于$yi$

由上面的定义我们可以知道，只要计算出第3步中的各个条件概率，就可以判断$x$属于哪一类。计算公式为：
$$
p(yi|x)=p(x|yi)p(yi)/p(x)
$$
而所有的$p(x)$是相同的，只需要比较$p(x|yi)p(yi)$的大小即可，对于$p(yi)$我们可以通过给定的训练集（已知分类）直接求出来，最重要的是求解$p(x|yi)$，由于朴素贝叶斯加上每个属性相互独立，因此其求解可转换为：
$$
p(x|yi)=p(a1|yi)p(a2|yi)...p(am|yi)
$$
需要注意的是当属性为连续型的时候，可以假设$p(aj|yi)$服从正态分布$N(u_{y_i,j},\sigma_{y_i,j}^2)$，其中$u_{y_i,i}$和$\sigma_{y_i,i}^2$分别是第$yi$类样本在第$j(j\in[1,m])$个属性上取值的均值和方差，则有：
$$
p(x_i|y_i)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}exp(-\frac{(x_i-u_{c,i})^2}{2\sigma_{c,i}^2})
$$

## 2.3 CART决策树

决策树算法采用树形结构，使用层层推理来实现最终的分类。决策树由下面几种元素构成：

- 根结点(1个)：包含样本的全集
- 内部结点(若干个)：对应特征属性测试
- 叶结点：代表决策的结果

![2019-09-17-jiegou](https://i.loli.net/2020/09/30/x6TJEqlNGDmRgkp.jpg)

预测时，在树的内部节点处用某一属性值进行判断，根据判断结果决定进入哪个分支节点，直到到达叶节点处，得到分类结果。

决策树学习的重点包括：

1. 如何选择最优划分属性，方法是在计算出来的各个特征的各个值的基尼系数中，选择基尼系数最小的特征A及其对应的取值a作为最优特征和最优切分点。 然后根据最优特征和最优切分点，将本节点的数据集划分成两部分$D_1$和$D_2$ ，同时生成当前节点的两个子节点，左节点的数据集为 $D_1$ ，右节点的数据集为$D2$
2. 剪枝策略，主要是为了防止过拟合，一般分为两种“预剪枝”和“后剪枝”(源码中为“后剪枝”)

三种典型的决策树算法为：

![v2-a8ae0794562f548e0b0e6f21733da7fa_1440w](https://i.loli.net/2020/09/30/5UXRZLqoxIKy9an.jpg)

本文采用的是CART算法，采用基尼系数来选择划分属性，基尼系数的定义如下：
$$
Gini(D)=1-\sum_{k=1}^{|y|}p_k^2
$$
其中$p_k$表示当前样本D中第$k$样本所占的比例$(k=1,2,...,{|y|})$。

由于Iris,Wine和Breast Cancer都是连续型数据因此采用了连续数学离散化技术，对于某个属性的所有取值都求一次基尼指数，选取属性中基尼系数最小属性值的作为划分点。

# 三.参考博客

[机器学习（一）——K-近邻（KNN）算法](https://www.cnblogs.com/ybjourney/p/4702562.html)

[[R]Iris朴素贝叶斯分类案例](https://zhuanlan.zhihu.com/p/29431399)

[CART决策树(Decision Tree)的Python源码实现](https://zhuanlan.zhihu.com/p/32164933)

[决策树 – Decision tree](https://easyai.tech/ai-definition/decision-tree/)

[决策树算法--CART分类树算法](https://zhuanlan.zhihu.com/p/139523931)

