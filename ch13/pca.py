'''
    ch13 利用PCA来简化模块
'''
from numpy import *

'''
    p246--13.1 PCA算法
    加载数据的方法
    params: fileName：文件名
        delim: 拆分数据的标记
    return：拆分完成的数据
'''
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

'''
    p246--13.1 PCA算法
    params: dataMat：数据集
        topNfeat：应用的N个特征
    return: lowDDataMat: 低维的数据集
        reconMat：重构后的数据
'''
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals   #去平均值
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    # 从小到大对N个值排序开始
    eigValInd=eigValInd[:-(topNfeat):-1]
    redEigVects = eigVects[:,eigValInd]
    # 从小到大对N个值排序结束
    lowDDataMat = meanRemoved * redEigVects   # 将数据转换到新的空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

'''
    测试pca方法
'''
def testPca():
    import matplotlib
    import matplotlib.pyplot as plt

    dataMat = loadDataSet('testSet.txt')
    lowDDat, reconMat = pca(dataMat, 2)
    print('shape(lowDDat):', shape(lowDDat))

    #绘制图形
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    fig.show()

'''
    p248--13.2将NaN替换成平均值的函数
'''
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0], i])  #计算所有非NaN的平均值

        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal   #将所有的NaN置为平均值
    return datMat

'''
    p249--测试NaN函数
'''
def testReplaceNaNWithMean():
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)  #计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))
    print('eigVals的值如下: \n', eigVals)

'''
    内部的测试方法
'''
if __name__=="__main__":
    #testPca()    #测试pca方法
    testReplaceNaNWithMean()    #测试替换方法