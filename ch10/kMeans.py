'''
    ch10-K均值的代码
'''
from numpy import *

'''
    p186--10.1 k均值聚类支持函数
    加载文件的方法
    params: fileName: 文件名
    return：dataMat：数据集 
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  #这里与课本不同，与上一章一样的进行修改
        dataMat.append(fltLine)
    return dataMat

'''
    p186--10.1 k均值聚类支持函数
    计算欧氏距离
'''
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

'''
    p186--10.1 k均值聚类支持函数
    为给定的数据集构建一个包含k个随机质心的集合
'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):      #构建簇质心
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

'''
    p187--测试随机质心的方法
'''
def testRandCent():
    dataMat = mat(loadDataSet('testSet.txt'))
    print('所有的数据：\n', dataMat)
    print('min(dataMat[:, 0]： ',  min(dataMat[:, 0]))

'''
    p187--10.2-k均值聚类算法
    params: dataSet:数据集
        k: 簇的数目
        distMeas: 距离的计算方法，默认为欧氏距离
        createCent: 质心类型，默认为随机的质心 
    return:
'''
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    #参数：dataset,num of cluster,distance func,initCen
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))#store the result matrix,2 cols for index and error
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#for every points
            minDist = inf;minIndex = -1#init
            for j in range(k):#for every k centers，find the nearest center
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:#if distance is shorter than minDist
                    minDist=distJI;minIndex=j# update distance and index(类别)
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                #此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
            clusterAssment[i,:] = minIndex,minDist**2
        #print(centroids)
        # update k center
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

'''
    p189--测试kMeans方法
'''
def testKMeans():
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clusAssing = kMeans(dataMat, 4)
    print('myCentroids: \n', myCentroids)
    print('clusAssing: \n', clusAssing)

'''
    p191--二分K均值聚类算法
    params: dataSet: 数据集
        k：要聚成的类簇的数量
        disMeas: 距离计算方法，默认欧氏距离
'''
def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error for each point
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf #init SSE
        for i in range(len(centList)):#for every centroid
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],:]  # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)# k=2,kMeans
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE: #judge the error
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #new cluster and split cluster
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment

'''
    p192--测试二分聚类方法
'''
def testBiKmeans():
    dataMat = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat,3)
    print('centList: \n', centList)

'''
    p194--10.4--Yahoo!PlaceFinder API
'''
import urllib
import json

'''
    获取地理位置
    params: stAddress: 静态位置
        city: 城市
    return: json数据
'''
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)   #与课本不同，原因是python3与python2的区别，参考地址：https://www.cnblogs.com/RUI-Z/p/8617409.html
    yahooApi = apiStem + url_params  # print url_params
    print
    yahooApi
    c = urllib.request.urlopen(yahooApi)   #这里与课本不同，原因是python3的urlopen在request下，参考：https://blog.csdn.net/u010899985/article/details/79595187
    return json.loads(c.read())


from time import sleep

'''
    查找批量位置的方法
    params: fileName：文件名称
    return: 
'''
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

'''
    p195--geo位置测试
'''
def testGeo():
    geoResults = geoGrab('1 VA Center', 'Augusta, ME')

'''
    p196--球面距离计算及簇绘图函数
    params: vecA:点1的经纬度
        vecB：点2的经纬度
    return: 两个点间的距离
'''
def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt

'''
    p196--10.5-球面距离计算及簇绘图函数
    簇绘图函数
'''
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']    #聚类记号
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  #基于图像创建矩阵
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


'''
    模块内部的测试
'''
if __name__ == '__main__':
    #pass
    #testRandCent()     #测试随机质心的方法
    #testKMeans()    #测试kMeans方法
    #testBiKmeans()  #测试二分聚类方法
    #testGeo()    #测试Geo位置
    clusterClubs(5)   #测试绘制功能