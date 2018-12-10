'''
    20181210 02:07决策树
'''

from math import log
import operator   #为p40代码增加的头文件
import matplotlib.pyplot as plt   #为p43代码增加的头文件

#全局变量区域开始
decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 定义文本框和箭头格式
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")
#全局变量区域结束

'''
    3.1 计算给定数据集的香农熵——P36
    计算给定数据集的香农熵
    param: dataSet 数据集
    return: shannonEnt
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    #print('对象个数：%d' % numEntries)
    labelCounts = {}
    for featVec in dataSet:    #为所有可能分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #print('标签的数量：%s' % labelCounts)
    for key in labelCounts:
        #print(labelCounts[key])
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)   #以2为底的对数
    return shannonEnt

'''
    p36的测试代码
    创建鱼签定数据的方法
    return: dataSet数据集, labels输出标记集
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
    p37——3.2按照给定的特征划分数据集
    param:dataSet：数据集
        axis：数轴
        value：值
    return：retDataSet：返回数据集
'''
def splitDatSet(dataSet, axis, value):
    retDataSet = []   #创建新的list对象
    for featVec in dataSet:   #将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
    p38--3.3 选择最好的数据集划分方式
    param: dataSet：数据集
    return bestFeature：返回特征
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]   #创建唯一的分类标签列表
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:   #计算每种划分方式的信息熵
            subDataSet = splitDatSet(dataSet, 1, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):   #计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
    p40--投票表决代码函数
    param: classList：分类列表
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
    p41--创建树的函数代码
    param: dataSet：数据集
        labels：标签
    return myTree 树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  #类别相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  #遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]    #得到列表包含的所有属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDatSet(dataSet, bestFeat, value), subLabels)

    return myTree

'''
    p43--3.5 使用文本注释绘制树结点
    param: nodeTxt：节点文本
        centerPt：中心点
        parentPt：父节点
        nodeType：节点类型
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #绘制带箭头的注释
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                          xytext=centerPt, textcoords='axes fraction', \
                          va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

'''
    p43--3.5 使用文本注解绘制树节点

def createPlot():
    decisionNode = dict(boxstyle = 'sawtooth', fc='0.8')    #定义文本框和箭头格式
    leafNode = dict(boxstyle='round4', fc='0.8')

    createPlot.axl = plt.subplot(111, frameon=False)

    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
'''

'''
    p45--3.6获取叶节目的数目和树的层数
    param: myTree:树实例
'''
def getNumLeafs(myTree):
    numLeafs = 0
    #print('键的值2：%s' % myTree.keys())
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():   #测试节点的数据类型的是否为字典
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

'''
    p45--3.6获取叶节目的数目和树的层数
    param: myTree:树实例
'''
def getTreeDepth(myTree):
    #print('键的值：%s' % myTree.keys())
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

'''
    p45--3.6获取叶节目的数目和树的层数
    param: myTree:树实例
'''
def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]

'''
    p46--3.7 plotTree函数：在父子节点间添加文本信息
    param: cntPt:中心点
        parentPt:父节点
        txtString: 字符串
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)

'''
    p46--3.7 plotTree函数：在父子节点间添加文本信息
    param: cntPt:中心点
        parentPt:父节点
        nodeTxt: 字符串
'''
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)   #计算宽
    depth = getTreeDepth(myTree)     #计算高
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yoff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD    #减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

'''
    创建图形
    param: inTree输入树
'''
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.axl = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

'''
    p49--3.8 使用决策树的分类函数
    params: inputTree:输入树
        featLabels:标签值
        testVec:测试向量
'''
def classify(inputTree, featLabels, testVec):
    #print(inputTree)
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

'''
    p50--3.9使用pickle模块存储决策树
    params:inputTree：输入树
        fileName：文件名
'''
def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(fileName):
    import pickle
    fr = open(fileName, 'rb')
    return pickle.load(fr)

'''
    p51 隐形眼镜数据集测试方法
'''
def testHideGlasses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescirpt', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print('lensesTree:%s' % lensesTree)
    createPlot(lensesTree)

'''
    完成主函数的编写
'''
if __name__ == '__main__':
    myDat, labels = createDataSet()
    print('数据集：%s，标签集:%s' % (myDat, labels))
    #print('使用最好的特征进行划分后的结果：%s' % chooseBestFeatureToSplit(myDat))
    #print('划分后的数据集：%s' % myDat)
    #print('分解后的数据集：%s' % splitDatSet(myDat, 0, 1))
    #print('香农熵：%s' % calcShannonEnt(myDat))

    #myDat[0][-1] = 'maybe'
    #print('修改数据后的数据集：%s' % myDat)
    #print('修改数据后的香农熵：%s' % calcShannonEnt(myDat))

    #生成树信息
    #myTree = createTree(myDat, labels)
    #print('生成的树信息为：%s' % myTree)

    #createPlot()    #打印图形

    '''
    #print(retrieveTree(1))    #追踪
    myTree = retrieveTree(0)    #追踪
    print('生成的数据：%s' % myTree)
    createPlot(myTree)
    print('[1,0]的分类标签为：%s' % classify(myTree, labels, [1,0]))
    print('[1,1]的分类标签为：%s' % classify(myTree, labels, [1, 1]))
    #print('键的值1：%s' % myTree.keys())
    print('叶子节点数量：%s' % getNumLeafs(myTree))
    print('树的深度为：%s' % getTreeDepth(myTree))

    #使用pickle存储与读取决策树开始
    storeTree(myTree, 'classifierStorage1.txt')
    grabTree('classifierStorage1.txt')
    # 使用pickle存储与读取决策树结束
    '''

    #调用隐形眼镜数据测试方法
    testHideGlasses()
