'''
    ch14利用Python实现SVD
'''
from numpy import *
from numpy import linalg as la

'''
    p255--14.3 利用Python实现SVD
    第一个测试函数
'''
def test1():
    U, Sigma, VT = linalg.svd([[1,1],[7,7]])
    print('U：\n', U)
    print('Sigma：', Sigma)
    print('VT：\n', VT)

'''
    p256--加载特定数据的方法
'''
def loadExData():
    return[[1,1,1,0,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]

'''
    p256--测试特定数据的SVD结果
'''
def testLoadExDataSVD():
    data = loadExData()
    U, Sigma, VT = linalg.svd(data)
    print('Sigma：\n', Sigma)   #一共有5个值，前3个值与课本一致，后面的2个值在不同的机器上运算结果不同

    #继续添加14.3节后面的代码
    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print('U[:,:3]*Sig3*VT[:3,:]的值为：\n', U[:,:3]*Sig3*VT[:3,:])

'''
    p259--相似度计算
'''
# 欧氏距离
def eulidSim(inA, inB):
    return 1.0/(1.0+la.norm(inA-inB))

# 皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]

#余弦相似度
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

#测试余弦相似度
def testCosSim():
    myMat = mat(loadExData())
    print('欧氏距离1：', eulidSim(myMat[:,0], myMat[:,4]))
    print('欧氏距离2：', eulidSim(myMat[:,0], myMat[:,0]))

    print('余弦相似度1：', cosSim(myMat[:,0], myMat[:,4]))
    print('余弦相似度2：', cosSim(myMat[:,0], myMat[:,0]))

    print('皮尔逊相关系数1：', pearsSim(myMat[:,0], myMat[:,4]))
    print('皮尔逊相关系数2：', pearsSim(myMat[:,0], myMat[:,0]))

'''
    p261--14.2基于物品相似度的推荐引擎
    标准估计函数
    params: dataMat：数据
        user：用户
        simMeas:
        item：
    return: 相似率
'''
def standEst(dataMat, user, simMeas, item):#数据矩阵、用户编号、相似度计算方法和物品编号
    n = shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        #寻找两个用户都做了评价的产品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:#存在两个用户都评价的产品 计算相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity #计算每个用户对所有评价产品累计相似度
        ratSimTotal += similarity * userRating  #根据评分计算比率
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

'''
    p261--14.2基于物品相似度的推荐引擎
    使用SVD的推荐引擎
    params: dataMat：数据
        user：用户
        N：
        simMeas:
        estMethod：会计方法，默认为上面的标准估计
    return: 排序后的数据
'''
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1] #寻找用户未评价的产品
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)#基于相似度的评分
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

'''
    p262--测试方法
'''
def testRecommend():
    myMat = mat(loadExData())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print('myMat的内容如下：\n', myMat)
    print('默认推荐内容如下：\n', recommend(myMat, 2))
    print('欧氏估计的推荐内容如下：\n', recommend(myMat, 2, simMeas=eulidSim))
    print('皮尔逊估计推荐内容如下：\n', recommend(myMat, 2, simMeas=pearsSim))

'''
    p264--14.3 基于SVD的评估分析
    params: dataMat：数据
        user：用户
        simMeas:
        item：
    return: 相似率
'''
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat) #不同于stanEst函数，加入了SVD分解
    Sig4 = mat(eye(4) * Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I #降维：变换到低维空间
    #下面依然是计算相似度，给出归一化评分
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

'''
    p265--测试基于SVD的评估分析方法
'''
def testSvdEst():
    myMat = mat(loadExData())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print('使用svd进行相似度计算的结果：\n', recommend(myMat, 1, estMethod=svdEst))
    print('添加皮尔逊估计进行相似度计算的结果：\n', recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

'''
    p266--14.6基于SVD的图像压缩
    打印矩阵。由于矩阵包含了浮点数,因此必须定义浅色和深色
'''
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end='')   #python2，只要使用print 'hello',最后加个逗号，则不会自动换行
            else:
                print(0, end='')   #python中，使用end=''，则可以不自动换行，继续前面的内容打印
        print('')

'''
    p266--14.6基于SVD的图像压缩
    压缩图像
'''
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat) #SVD分解
    SigRecon = mat(zeros((numSV, numSV))) #创建初始特征
    for k in range(numSV):#构造对角矩阵
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

'''
    p267--测试压缩方法
'''
def testCompress():
    imgCompress(2)

'''
    内部测试类
'''
if __name__ == "__main__":
    #pass
    #test1()    #第一个测试函数
    #testLoadExDataSVD()   #测试特定数据的SVD
    #testCosSim()    #测试cosSim函数
    #testRecommend()    #测试预测信息
    #testSvdEst()    #测试基于SVD的预测
    testCompress()   #测试图像压缩的方法