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
    内容测试主函数
'''
if __name__ == '__main__':
    #testLoadData()   #测试加载数据的方法
    testSmoSample()   #测试smoSample