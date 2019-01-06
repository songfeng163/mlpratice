'''
    ch09 regTrees文件的内容
'''
from numpy import *

'''
    p161--9.1 载入数据
    CART算法的实现代码
    params: fileName: 文件名称
    return: dataMat: 数据矩阵
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # python3不适用：fltLine = map(float,curLine) 修改为：
        fltLine = list(map(float, curLine))  #将每行映射成浮点数，python3返回值改变，所以需要
        dataMat.append(fltLine)
    return dataMat

'''
    p162--9.1 切分数据集为两个子集
    CART算法的实现代码
    params: dataSet: 数据集
        feature: 特征
        value: 特征值
    return: dataMat: 数据矩阵
'''
def binSplitDataSet(dataSet, feature, value): #数据集 待切分特征 特征值
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    #下面原书代码报错 index 0 is out of bounds,使用上面两行代码
    #mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    #mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


'''
    p164--程序9.2开始
    回归树的切分函数
'''
#Tree结点类型：回归树
def regLeaf(dataSet):#生成叶结点，在回归树中是目标变量特征的均值
    return mean(dataSet[:,-1])
#误差计算函数：回归误差
def regErr(dataSet):#计算目标的平方误差（均方误差*总样本数）
    return var(dataSet[:,-1]) * shape(dataSet)[0]
#二元切分
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #切分特征的参数阈值，用户初始设置好
    tolS = ops[0] #允许的误差下降值
    tolN = ops[1] #切分的最小样本数
    #若所有特征值都相同，停止切分
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#倒数第一列转化成list 不重复
        return None,leafType(dataSet)  #如果剩余特征数为1，停止切分1。
        # 找不到好的切分特征，调用regLeaf直接生成叶结点
    m,n = shape(dataSet)
    S = errType(dataSet)#最好的特征通过计算平均误差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #遍历数据的每个属性特征
        # for splitVal in set(dataSet[:,featIndex]): python3报错修改为下面
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):#遍历每个特征里不同的特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#对每个特征进行二元分类
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:#更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果切分后误差效果下降不大，则取消切分，直接创建叶结点
    if (S - bestS) < tolS:
        return None,leafType(dataSet) #停止切分2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #判断切分后子集大小，小于最小允许样本数停止切分3
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue#返回特征编号和用于切分的特征值
'''
    p164--程序9.2开始
    回归树的切分函数
'''

'''
    p162--构建tree
    params: dataSet: 数据集
        leafType: 叶子类型，默认为regLeaf
        errType: 错误类型，默认为regErr
        ops： 是一个包含树所需其他参数的元组
    return: retTree： 构建好的树
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #数据集默认NumPy Mat 其他可选参数【结点类型：回归树，误差计算函数，ops包含树构建所需的其他元组】
    feat,val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val #满足停止条件时返回叶结点值
    #切分后赋值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #切分后的左右子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

'''
    p162--测试创建树的函数
'''
def testCreateTree():
    testMat = mat(eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('mat0:\n %s \nmat1: \n %s\n' % (mat0, mat1))

    myDat = loadDataSet('ex00.txt')
    myMat = mat(myDat)
    regTrees = createTree(myMat)
    print('regTrees: \n%s' % regTrees)

    myDat1 = loadDataSet('ex0.txt')
    myMat1 = mat(myDat1)
    regTrees1 = createTree(myMat1)
    print('regTrees1: \n%s' % regTrees1)

    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    regTrees2 = createTree(myMat2, ops=(10000,4))  #修改生成树的参数，减少节点数量
    print('regTrees2: \n%s' % regTrees2)


'''
    p169--9.3 回归树剪枝函数开始
'''

#判断输入是否为一棵树
def isTree(obj):
    return (type(obj).__name__=='dict') #判断为字典类型返回true
#返回树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


#树的后剪枝
def prune(tree, testData):#待剪枝的树和剪枝所需的测试数据
    if shape(testData)[0] == 0: return getMean(tree)  # 确认数据集非空
    #假设发生过拟合，采用测试数据对树进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])): #左右子树非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #剪枝后判断是否还是有子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #判断是否merge
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #如果合并后误差变小
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


'''
    测试剪枝函数效果
'''
def testPrune():
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    myTree = createTree(myMat2, ops=(0, 1))  # 以每个数据为节点，生成叶子，生成最大树
    print('myTree: \n%s' % myTree)

    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDatTest)
    returnTree = prune(myTree, myMat2Test)
    print('returnTree: \n%s' % returnTree)

'''
    p169--9.3 回归树剪枝函数结束
'''


'''
    p172--9.4 模型树叶节点生成函数开始
'''
#模型树
def linearSolve(dataSet):   #将数据集格式化为X Y
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0: #X Y用于简单线性回归，需要判断矩阵可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#不需要切分时生成模型树叶节点
    ws,X,Y = linearSolve(dataSet)
    return ws #返回回归系数

def modelErr(dataSet):#用来计算误差找到最佳切分
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

#测试上面的几个函数的方法
def testModeLeaf():
    myMat2 = mat(loadDataSet('exp2.txt'))
    regTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    print('regTree: \n%s' % regTree)

'''
    p172--9.4 模型树叶节点生成函数结束
'''


'''
    p174--9.5 用树回归进行预测的代码开始
'''
#用树回归进行预测
#1-回归树
def regTreeEval(model, inDat):
    return float(model)
#2-模型树
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)
#对于输入的单个数据点，treeForeCast返回一个预测值。
def treeForeCast(tree, inData, modelEval=regTreeEval):#指定树类型
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):#有左子树 递归进入子树
            return treeForeCast(tree['left'], inData, modelEval)
        else:#不存在子树 返回叶节点
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
#对数据进行树结构建模
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
#测试线性回归效果，与树回归对比
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# p175的测试代码
def testForeCast():
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    myRate = corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]
    print("myRate: ", myRate)

    myTree1 = createTree(trainMat, modelLeaf, modelErr, (1, 20))  #这里的参数与上面不同
    yHat1 = createForeCast(myTree1, testMat[:, 0], modelTreeEval)  #这里的参数与上面不同，函数加()表示要得到返回结果，不加()表示对函数的调用
    myRate1 = corrcoef(yHat1, testMat[:, 1], rowvar=0)[0,1]
    print("myRate1: ", myRate1)

    ws, X, Y = linearSolve(trainMat)
    print('ws: \n %s' % ws)
    #print('\nX: \n%s' % X)
    #print('\nY: \n%s' % Y)

    #循环执行
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    myRate2 = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print("myRate2: %s" % myRate2)

'''
    p174--9.5 用树回归进行预测的代码结束
'''

'''
    文件内部的测试函数
'''
if __name__ == '__main__':
    #testCreateTree()  #测试创建树的方法
    #testPrune()    #测试剪枝函数的方法
    #testModeLeaf()   #测试模型树结点
    testForeCast()    #测试预测函数功能