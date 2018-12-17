'''
    第5章 logistic回归
'''
from numpy import *

'''
    p78--5.1 Logistic回归梯度上升优化函数
    加载文件中的数据集的方法
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''
    p78--5.1 Logistic回归梯度上升优化函数
    sigmoid函数
'''
def sigmoid(inX):
    return 1.0/(1+ exp(-inX))

'''
    p78--5.1 Logistic回归梯度上升优化函数
    计算梯度下降
'''
def gradAscent(dataMatIn, classLabels):
    #转换为矩阵类型开始
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    #转换为矩阵类型结束

    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))

    #矩阵相乘开始
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    #矩阵相乘结束
    return weights

'''
    p79 Logistic回归梯度上升优化函数
    梯度下降的测试
'''
def testGradAscent():
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print('weights的值如下：\n%s' % weights)

    #p79-80 添加对输出图形的测试
    plotBestFit(weights.getA())    #getA()把矩阵转换为一个ndarray

'''
    p79--5.2 画出数据集和Logistic回归最佳拟合直线的函数
'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/ weights[2]   #最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

'''
    p81--5.3 随机梯度上升算法
'''
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)   #得到行与列表
    print('行数：%d,列数：%d' % (m, n))
    alpha = 0.01
    weights = ones(n)   #定义初始化权重为所有元素为1的矩阵
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
    p81 stocGradAscent随机梯度上升优化函数
    梯度下降的测试
'''
def testStocGradAscent0():
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(array(dataArr), labelMat)
    print('随机梯度上升的weights的值如下：\n%s' % weights)
    plotBestFit(weights)

'''
    p82 --5.4 改进的随机梯度上升算法
'''
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)  # 得到行与列表
    weights = ones(n)  # 定义初始化权重为所有元素为1的矩阵
    for j in range(numIter):
        dataIndex = list(range(m))   #range不是数组对象，要转换成list对象
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01   #alpha每次迭代时需要调整
            randIndex = int(random.uniform(0, len(dataIndex)))  #随机选择更新
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
    p84 stocGradAscent1改进随机梯度上升优化函数
    梯度下降的测试
'''
def testStocGradAscent1():
    dataArr, labelMat = loadDataSet()
    #weights = stocGradAscent1(array(dataArr), labelMat)   #采用默认的150次
    weights = stocGradAscent1(array(dataArr), labelMat, 500)  # 修改默认值为500次
    print('随机梯度上升的weights的值如下：\n%s' % weights)
    plotBestFit(weights)

'''
    p86--5.5 Logistic回归分类函数
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:    #sigmoid值>0.5就预测标签为1, 否则为0
        return 1.0
    else:
        return 0.0

'''
    疝病测试
'''
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():  #使用readlines()读取所有的行数据
        currLine = line.strip().split('\t')   #使用tap标记划分数据
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))    #获取行数据，读取分离的每个数据，加入行中
        trainingSet.append(lineArr)   #获取训练数据
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0     #错误数
    numTestVec = 0.0   #测试的向量数
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('The error rate of this test is : %f' % errorRate)
    return errorRate

'''
    p87--多次调用colicTest进行计算的方法
'''
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))

'''
    模块内部测试方法
'''
if __name__ == '__main__':
    #testGradAscent()   #测试梯度上升算法
    #testStocGradAscent0()    #测试随机梯度上升算法
    #testStocGradAscent1()  # 测试改进的随机梯度上升算法
    multiTest()    #调用多次疝病测试函数