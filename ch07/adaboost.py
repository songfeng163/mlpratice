'''
    20190102 创建
    机器学习实践：第七章代码
'''

from numpy import *

'''
    p119--构造数据
    生成样本与标签的函数
'''
def loadSimpleData():
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

'''
    测试生成数据的函数
'''
def testData():
    datMat, classLabels = loadSimpleData()
    print('样本为：%s\n' % datMat)
    print('标签为：%s' % classLabels)

'''
    p120--7.1
    单层决策树生成函数：通过阈值比较进行分类
    params: dataMatrix：输入数据
        dimen：维数
        threshVal: 阈值
        threshIneq：不相等的值
    return: retArray：返回的数组
        
'''
def stumpClassify(dataMatrix, dimen, thresshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= thresshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > thresshVal] = 1.0
    return retArray

'''
    p120--7.1
    单层决策树生成函数：遍历stumpClassify()函数所有的可能输入值，找到数据集上最佳的单层决策树
    params: dataArr： 输入数据
        classLabels：分类标签
        D: 权重向量D
    return: bestStump：最好的树桩
        minError：最小误差
        bestClassEst：最好的分类估计
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)  #取出行与列的数据来
    numSteps = 10.0  #最小步长，默认为10
    bestStump = {}   #空的字典
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 最小误差为无穷大

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()  #所有行的第i+1列作为最小值
        rangeMax = dataMatrix[:, i].max()  #所有行的第i+1列作为最大值
        stepSize = (rangeMax - rangeMin)/numSteps  #步长
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  #计算机加权错误率
                #print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' % \
                #      (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

'''
    p121--测试代码，宋锋更新
    测试生成树桩函数，了解生成过程
'''
def testBuildStump():
    datMat, classLabels = loadSimpleData()
    D = mat(ones((5,1))/5)   #生成每个元素的值为0.2的5行1列的向量
    print("D的值：", D)
    bestStump, minError, bestClassEst = buildStump(datMat, classLabels, D)
    print('bestStump: %s\n' % bestStump)
    print('minError: %s' % minError)
    print('bestClassEst: %s' % bestClassEst)

'''
    p122--7.2
    基于单层决策树的AdaBoost训练过程
    DS: decision stump：单层决策树
    params: dataArr：输入数据
        classLabels: 分类标签
        numIt: 循环次数
    return: weakClassArr：弱分类数组
'''
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)   #生成每个元素的值为1/m的m行1列的向量，权重向量
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst: ', classEst.T)

        #为下一次迭代计算D开始
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        #为下一次迭代计算D结束

        #错误率累加计算开始
        aggClassEst += alpha * classEst
        #print('aggClassEst: ', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print('total error:', errorRate)
        #错误率累加计算结束

        if errorRate == 0.0:   #结束循环的条件
            break
    #return weakClassArr    #实例7.5以前的代码使用这个返回值
    return weakClassArr, aggClassEst  #为了程序7.5测试而使用这个返回值

'''
    p122--测试代码，宋锋更新
    测试adaBoost方法
'''
def testAdaBoost():
    datMat, classLabels = loadSimpleData()
    weakClassArr = adaBoostTrainDS(datMat, classLabels, 9)
    print('weakClassArr: ', weakClassArr)

'''
    p124--7.5测试算法
    基于AdaBoost的分类
    params: datToClass：要分类的数据
        classifierArr：分类数组
    return: 分类标签
'''
def adaClasify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]   #行数
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print('aggClassEst: ', aggClassEst)
    return sign(aggClassEst)

'''
    p125--宋锋修改：测试分类方法
'''
def testAdaClassify():
    datArr, labelArr = loadSimpleData()
    classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
    #myClassify = adaClasify([0,0], classifierArr)
    myClassify = adaClasify([[5, 5],[0, 0]], classifierArr)
    print('myClassify: \n%s' % myClassify)

'''
    p126-7.4--自适应数据加载函数
    param: fileName：文件名称
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))  #特征数量
    dataMat = []   #数据集
    labelMat = []  #标签集
    fr = open(fileName)   #文件读取器
    for line in fr.readlines():  #进行行遍历
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
    p126--宋锋修改测试方法
'''
def testHorseData():
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArry = adaBoostTrainDS(datArr, labelArr, 50)
    print("classifierArry: \n%s" % classifierArry)
    print('测试数据的测试结果如下：\n')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClasify(testArr, classifierArry)
    errArr = mat(ones((67, 1)))
    errNum = errArr[prediction10!=mat(testLabelArr).T].sum()
    print("错误数：%s，错误率：%s" % (errNum, errNum/67))

'''
    p130--7.5 ROC 曲线的绘制及AUC计算函数
'''
def plotROC(predStrenths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClsas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClsas)
    xStep = 1/float(len(classLabels)-numPosClsas)
    sortedIndicies = predStrenths.argsort()     #获取排好序的索引
    fig = plt.figure()  #生成图像
    fig.clf()   #清空图像
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')  #蓝色绘图
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')  #蓝色虚线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    ax.axis([0,1,0,1])
    plt.show()   #显示图形
    print('the Area Under the Curve is: ', ySum * xStep)

'''
    p131--宋锋修改打印图形的方法
'''
def testPlotFig():
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)

'''
    模块内部测试方法
'''
if __name__ == '__main__':
    #testData()   #测试生成数据
    #testBuildStump()   #测试生成树桩的函数
    #testAdaBoost()     #测试adaBoost方法
    #testAdaClassify()  #测试adaClassify分类方法
    #testHorseData()     #测试马类数据
    testPlotFig()    #打印图形