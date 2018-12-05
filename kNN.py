#机器学习实践

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties #字体管理器  #参考文献：https://blog.csdn.net/asialee_bird/article/details/81027488
import os    #加载操作系统类

#设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

pathNow=os.path.abspath('.')   # 表示当前所处的文件夹的绝对路径
pathParent=os.path.abspath('..')   # 表示当前所处的文件夹父文件夹的绝对路径


'''
    20181206完成
    对应p17的代码
    2.0 创建数据集的方法
    无参数，返回group,labels
'''
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

'''
 20181205完成
 对应p19的代码
 2.1 k近邻算法
 参数：inX:用于分类的输入向量
    dataSet：输入的训练样本
    labels：标签向量
    k：表示用于选择的最近邻数目
'''
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]   #shape[0]行数，shape[1]列数,结果是一个数
    # (dataSet, 1)，用行数和1组成一个元组，表示dataSetSize行，1列
    # tile(intX, (dataSet, 1))将intX扩展成dataSetSize行，1列的一个向量数据
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2    #数据平方
    sqDistnace = sqDiffMat.sum(axis=1)   #所有的值相加
    distance = sqDistnace ** 0.5    #将上一步的和开方
    sortedDistIndices = distance.argsort()   #对结果进行排序
    classCount = {}    #预定义一个数据
    for i in range(k):   #选择距离最小的k的点
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),   #python3中不再支持iteritems()，直接使用items
            key=operator.itemgetter(1), reverse = True)  #排序
    return sortedClassCount[0][0]

'''
    20181206完成——测试基础分类的方法
    对应p20的代码
'''
def testClassify0():
    group, labels = creatDataSet()
    print("创建的训练样本：")
    print(group)
    print('训练样本的标签：')
    print(labels)
    newSample = classify0([0, 0], group, labels, 3)
    print("测试样本所属于的分类：" + newSample)

'''
    20181206 完成
    对应p21的代码
    2.2 将文本记录转换为NumPy的解析程序
    文件的标签列为字符串时，使用这个方法
    输入参数： filename，文件名称
    返回：returnMat，返回的文件内容矩阵
         classLabelVector：分类标签
'''
def file2Matrix1(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  #将所有的数据放入数组
    numberOfLines = len(arrayOLines)  #文件的行数
    returnMat = zeros((numberOfLines, 3))   #创建返回的numpy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   #删除空白字符与回车字符
        listFromLine = line.split('\t')   #使用Tab键分割数据
        returnMat[index, :] = listFromLine[0:3]   #选取前3个数据存入返回的特征矩阵中去
        classLabelVector.append(listFromLine[-1])  #最后一列为分数，存入分类向量中去
        index += 1
    return returnMat, classLabelVector

'''
    20181206 完成
    对应p21的代码
    文件的标签列为数字时，使用这个方法
    2.2 将文本记录转换为NumPy的解析程序
    输入参数： filename，文件名称
    返回：returnMat，返回的文件内容矩阵
         classLabelVector：分类标签
'''
def file2Matrix2(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  #将所有的数据放入数组
    numberOfLines = len(arrayOLines)  #文件的行数
    returnMat = zeros((numberOfLines, 3))   #创建返回的numpy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   #删除空白字符与回车字符
        listFromLine = line.split('\t')   #使用Tab键分割数据
        returnMat[index, :] = listFromLine[0:3]   #选取前3个数据存入返回的特征矩阵中去
        classLabelVector.append(int(listFromLine[-1]))  #最后一列为分数，存入分类向量中去
        index += 1
    return returnMat, classLabelVector

'''
    20181206完成——文件到矩阵的功能测试
    对应p22的代码
'''
def testFile2Matrix():
    #调用datingTestSet.txt文件时，标签列为字符串，使用file2Matrix1方法
    #returnMat, classLabelVetor = file2Matrix1('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet.txt')
    returnMat, classLabelVetor = file2Matrix2('datingTestSet.txt')  # 直接使用相对路径
    # 调用datingTestSet2.txt文件时，标签列为数字，使用file2Matrix1方法
    #returnMat, classLabelVetor = file2Matrix2('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet2.txt')
    print('样本矩阵如下：')
    print(returnMat)
    print('样本分类标签如下：')
    print(classLabelVetor)

'''
    20181206完成——创建散点图
    对应p23-24的代码
'''
def createScatter():
    #调用datingTestSet.txt文件时，标签列为字符串，使用file2Matrix1方法
    #datingDataMat, datingLabels = file2Matrix1('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet.txt')
    # 调用datingTestSet2.txt文件时，标签列为数字，使用file2Matrix1方法
    #datingDataMat, datingLabels = file2Matrix2('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet2.txt')
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    fig = plt.figure()
    plt.xlabel(u'玩视频游戏所耗时间百分比', fontproperties=font)
    plt.ylabel(u'每周消费的冰淇淋公升数', fontproperties=font)
    ax = fig.add_subplot(111)
    #ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])  #普通的散点图

    #使用第2列和第3列的数据制作散点图
    #ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],  # 指定的散点图中点的颜色，要使用标签为数字的文件datingTestSet2.txt
    #           15.0 * array(datingLabels), 15.0 * array(datingLabels))

    #使用第1列和第2列的数据制作散点图，效果比上面更好
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],   # 指定的散点图中点的颜色，要使用标签为数字的文件datingTestSet2.txt
               15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

'''
    20181206添加带图例的散点图
    实现p24的图2-5效果
    参考文献：https://blog.csdn.net/xiaobaicai4552/article/details/79069207
'''
def createScatterWithLegend():
    #调用datingTestSet.txt文件时，标签列为字符串，使用file2Matrix1方法
    #datingDataMat, datingLabels = file2Matrix1('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet.txt')
    # 调用datingTestSet2.txt文件时，标签列为数字，使用file2Matrix1方法
    #datingDataMat, datingLabels = file2Matrix2('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet2.txt')
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.xlabel(u'每年获取的飞行常客里程数', fontproperties=font)
    plt.ylabel(u'玩视频游戏所耗时间百分比', fontproperties=font)
    axes = plt.subplot(111)

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []

    #注意：参考文献中的标签值为字符串，所以比较时用的“=='1'”等信息，而我这里的标签是数字，所以直接用数字比较即可
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    #颜色：r红，b蓝，g绿，k黑
    type1 = axes.scatter(type1_x, type1_y, s=20, c='r')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='b')
    type3 = axes.scatter(type3_x, type3_y, s=60, c='g')

    plt.legend((type1, type2, type3), ('不喜欢', '魅力一般', '极具魅力'))
    plt.show()

'''
    20181206完成
    对应p25的代码
    2.3 归一化特征值
    输入：dataSet 数据集
    返回： normDataSet归一化后的数据集
          ranges范围
          minVals最小值
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)   #最小值
    maxVals = dataSet.max(0)   #最大值
    ranges = maxVals - minVals   #范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]   #行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

'''
    20181206完成——p26的归一化测试代码
'''
def testAutoNorm():
    # 调用datingTestSet2.txt文件时，标签列为数字，使用file2Matrix1方法
    #datingDataMat, datingLabels = file2Matrix2('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet2.txt')
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print('归一化后的数据为：')
    print(normDataSet)
    print('数据范围为：')
    print(ranges)
    print('最小值为：')
    print(minVals)

'''
    20181206完成-p27的代码
    2.4 分类器针对约会网站的测试代码
'''
def datingClassTest():
    hoRatio = 0.08   #我的机器上0.08的值与书中0.10的值的计算结果类似，我的错误率是0.025
    #datingDataMat, datingLabels = file2Matrix2('D:\myprograms\PycharmProjects\mlpractice\ch02\datingTestSet2.txt')
    #datingDataMat, datingLabels = file2Matrix2(pathNow+'\datingTestSet2.txt')  #系统路径+相对路径
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVacs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVacs):
        classifierResult = classify0(normDataSet[i, :], normDataSet[numTestVacs:m, :], datingLabels[numTestVacs:m], 3)
        print('The classifier came back with: %d, the real answer is %d' % (classifierResult, datingLabels[i]))
        if(classifierResult!=datingLabels[i]):errorCount+=1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVacs)))

'''
    20181206完成-P28代码
    2.5 约会网站预测函数
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumes per year?'))
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normDataSet, datingLabels, 3)
    print('You will probably liked this person:', resultList[classifierResult - 1])

'''
    20181206完成-P28代码
    2.5 约会网站预测函数-中文
'''
def classifyPerson():
    resultList = ['不喜欢', '魅力一般', '极具魅力']
    percentTats = float(input('玩视频游戏所耗时间百分比?'))
    ffMiles = float(input('每年获取的飞行常客里程数?'))
    iceCream = float(input('每周消费的冰淇淋公升数?'))
    datingDataMat, datingLabels = file2Matrix2('datingTestSet2.txt')  # 直接使用相对路径
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normDataSet, datingLabels, 3)
    print('你这个人的可能的评价:', resultList[classifierResult - 1])

'''
    20181206完成——p29代码
    将图像 格式化为一个向量的的方法
    输入：要打开的文件路径+文件名
    返回：转换后的向量
'''
def img2vector(filename):
    returnVact = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVact[0, 32*i+j] = int(lineStr[j])
    return returnVact

'''
    20181206完成——p30面测试代码
'''
def testImg2Vector():
    testVector = img2vector('digits/testDigits/0_13.txt')
    print('第一行数据为：')
    print(testVector[0, 0:31])
    print('第二行数据为：')
    print(testVector[0, 32:63])

'''
    20181206完成——p30代码
    2.6 手写数字识别系统的测试代码
'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]   #从文件名中解析分类数字,这里直接是数字，不用转换，与课本不同
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]   #从文件名中解析分类数字,这里直接是数字，不用转换，与课本不同
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        #print(type(classNumStr))
        #print(type(classifierResult))
        print('the classifier came back with: %s, the real answer is: %s' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print('\nthe totle number of erros is : %d' % errorCount)
    print(type(errorCount/float(mTest)))
    print('\nthe total error rate is: %f' % (errorCount/float(mTest)))

'''
    20181206完成主函数的编写
'''
if __name__ == '__main__':
    #testClassify0()     #测试基础分类的方法
    #testFile2Matrix()   #测试文件转换矩阵的方法
    #createScatter()     #测试创建散点图的方法
    #createScatterWithLegend()  # 测试创建带有图例的散点图的方法
    #testAutoNorm()      #归一化测试
    #datingClassTest()   #p27页的整体测试代码
    #classifyPerson()    #p28 测试约会网站预测
    #testImg2Vector()    #测试图像到向量的转换
    handwritingClassTest()  #测试手写数字

