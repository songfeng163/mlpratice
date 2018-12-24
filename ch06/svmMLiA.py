'''
    20181220 第六章 支持向量机Support Vector Machine算法的实现
'''

from numpy import *
import random

'''
    p95--6.1
    SMO算法中的辅助函数
    param: fileName：文件名
    return: dataMat：数据集
        labelMat：标签集
'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():   #遍历读取每一行
        lineArr = line.strip().split('\t')   #使用tab标签划分数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])   #读取第1列与第2列数据
        labelMat.append(float(lineArr[2]))   #第3列数据为标签
    return dataMat, labelMat

'''
    p95--6.1
    随机选择数据
    params: i：第一个alpha下标
        m：alpha的数目
    return: j：选定的参数
'''
def selectJrand(i, m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))   #生成[0, m)中的一个实数
    return j

'''
    p95--6.1
    调整大于H或小于L的alpha值
    params: aj：要调整的值
        H: 上限High
        L: 下限Low
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
    p95--测试加载数据的方法
'''
def testLoadData():
    dataArr, labelArr = loadDataSet('testSet.txt')
    print('数据集dataSet如下：\n %s' % dataArr)
    print('结果集labelSet如下：\n %s' % labelArr)

'''
    p96--6.2 简化版SMO算法
    params: dataMatIn: 数据集
        classLabels: 类别标签
        C: 常数C
        toler: 容错率
        maxIter: 退出前最大的循环次数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))  #构建一个全部元素为0的alpha矩阵
    iter = 0   #循环计数变量
    while (iter < maxIter):
        alphaPairsChanged = 0   #记录alpha是否已经进行优化，每次循环中将这个值设置为0
        for i in range(m):     #在遍历中对参数的值进行调整
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b   #预测类别
            Ei = fXi - float(labelMat[i])  #误差
            # 如何alpha可以更改进入优化过程
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)   #随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                #保证alpha在0与C之间开始
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 保证alpha在0与C之间结束

                if L == H:
                    print('L == H')
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  #对i进行修改，修改量与j相同，但方向相反

                #设置常数项开始
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                     dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                    dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C >alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                #设置常数项结束

                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter =0
        print('Iteration number: %d' % iter)
    return b, alphas

'''
    p98--测试数据
'''
def testSmoSample():
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print('b=%s' % b)
    print('alphas>0的数据: \n%s' % alphas[alphas>0])

'''
    p107--6.6 核转换函数
'''
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


'''
    p100-6.3 完整版的Platt SMO的支持函数
    类的定义
'''
class optStruct:
    '''
    params: dataMatIn: 输入数据矩阵
        classLabels: 分类标签
        C: c参数
        toler:  容错率
    '''
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #误差缓存
        self.K = mat(zeros((self.m,self.m)))
        #p107增加的内容开始
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        # p107增加的内容结束

'''
    p100-6.3 完整版的Platt SMO的支持函数
'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


'''
    p100-6.3 完整版的Platt SMO的支持函数
    内循环中的的启发式方法
'''
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

'''
    p100-6.3 完整版的Platt SMO的支持函数
    更新期望值
'''
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

'''
    p101 -- 6.4 完整Platt SMO算法中的优化例程
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print( "j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #更新误差缓存
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

'''
    p102--6.5完整版的Platt SMO的外循环代码
    params: dataMatIn: 输入数据矩阵
        classLabels: 分类标签
        C: c参数
        toler:  容错率
        maxIter：退出前最大的循环次数
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False #切换整个设置循环
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

'''
    p103--测试Platt SMO的方法
'''
def testSmoP():
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)

'''
    p104--利用alpha值进行分类计算
'''
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

'''
    p104--测试进行分类的计算的方法
'''
def testCalcWs():
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    print('ws的值如下：\n %s' % ws)
    dataMat = mat(dataArr)
    print('dataMat[0]*mat(ws)+b = %s' % (dataMat[0]*mat(ws)+b))   #结果>0时属于1类，结果<0时属于-1类
    print('labelArr[0]=%s' % labelArr[0])
    print('dataMat[2]*mat(ws)+b = %s' % (dataMat[2] * mat(ws) + b))  # 结果>0时属于1类，结果<0时属于-1类
    print('labelArr[2]=%s' % labelArr[2])
    print('dataMat[1]*mat(ws)+b = %s' % (dataMat[1] * mat(ws) + b))  # 结果>0时属于1类，结果<0时属于-1类
    print('labelArr[1]=%s' % labelArr[1])

'''
    p108--6.8利用核函数进行分类的径向基测试函数
    param: k1--参数
'''
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 非常重要
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  # 构建支持向量矩阵
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

'''
    p111--6.9前导入的函数
    param: filename：文件名称
'''
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
    p111--6.9前导入的函数
    加载目录下的图片信息
'''
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

'''
    p112--测试数字信息
'''
def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('digits/trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('digits/testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


'''
    内容测试主函数
'''
if __name__ == '__main__':
    #testLoadData()   #测试加载数据的方法
    #testSmoSample()   #测试smoSample
    #testSmoP()   #测试Platt SMO的方法
    #testCalcWs()   #测试分类的方法
    #testRbf()     #测试利用核函数进行分类的径向基测试函数
    testDigits(('rbf', 20))   #调用测试数字的方法