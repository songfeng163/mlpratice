'''
    20181211第4章 基于概率论的分类方法：朴素贝叶斯
'''

from numpy import *
import feedparser

'''
    p58--4.1词表到向量的转换函数
'''
def loadDataSet():
    postingList=[['my','dog','has','flea','ploblems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList, classVec

'''
    p59--4.1创建一个包含在所有文档中出现的不重复词的列表
    param: dataSet：数据集
'''
def createVocabList(dataSet):
    vocabSet = set([])   #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''
    p59--4.1单词到向量的转换
    params: vocabList:单词列表
        inputSet：输入集合
'''
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)   #创建一个其中所有含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

'''
    p59--测试贝叶斯函数的方法
'''
def testBayes():
    listOPosts, listClases = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print('myVocablist的内容如下：\n%s' % myVocabList)
    print('words2Vec[0]的结果：\n%s' % setOfWord2Vec(myVocabList, listOPosts[0]))
    print('words2Vec[3]的结果：\n%s' % setOfWord2Vec(myVocabList, listOPosts[3]))

'''
    p61--4.2 朴素贝叶斯分类器训练函数
    params: trainMatrix: 训练的矩阵
        trainCategory: 训练的类型标签
    return: p0Vect, p1Vect, pAbusive
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom    #change to log()
    p0Vect = p0Num / p0Denom    #change to log()
    return p0Vect, p1Vect, pAbusive

'''
    p61--测试贝叶斯分类器训练函数的代码
'''
def testTrainNB0():
    listOPosts, listClasses = loadDataSet()
    createVocabList(listOPosts)
    myVocabList = createVocabList(listOPosts)
    trainMat= []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print('pAb=%s' % pAb)
    print('p0V is as follow: \n%s' % p0V)
    print('p1V is as follow: \n%s' % p1V)

'''
    p63--朴素贝叶斯分类函数
    params: vec2Classify：分类标签向量
        p0Vec：向量1
        p1Vec: 向量2
        pClass1:分类
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #向量中的对应元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

'''
    p61--测试朴素贝叶斯分类的方法
'''
def testingNB():
    listOPosts, listClasses = loadDataSet()  #加载数据
    myVocabList = createVocabList(listOPosts)   #生成词向量
    trainMat = []   #定义空的训练矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    #testEntry = ['love', 'my', 'dalmation']
    testEntry = ['dog', 'buying', 'my', 'dalmation']  #要测试的实体1
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']   #要测试的实体2，这里可以添加一个'silly',程序还可以发现它并不在词汇表中
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, "classified as：", classifyNB(thisDoc, p0V, p1V, pAb))

'''
    p64--文档词袋模型
    params: vocabList: 词列表
        inputSet：输入集
    return: returnVec 返回向量
'''
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
    p66--文件解析及完整的垃圾邮件测试函数
    param: bigString: 大字符串
'''
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
'''
    p66--检测垃圾邮件的方法
'''
def spamTest():
    docList = []
    classList = []
    fullText = []
    #导入并解析文件开始
    #将文本内容解析为词列表
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #导入并解析文件结束

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))   #本例中共有50个邮件文件   #在python3中要，使用list，否则运行时会出错
    testSet = []
    #随机构建训练集开始
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])  #将选择的对象添加到测试集
        del(trainingSet[randIndex])   #将选择的对象从训练集中删除
    #这里运行一次是一欠迭代，若要得到更加准确的分类器错误率，可以进行多次迭代，并求出错误率的平均值
    #随机构建训练集结束
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:   #遍历训练集，对每个邮件使用setOfWord2Vec方法构建词向量
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount)/len(testSet)   #错误率
    print('the error rate is: ', errorRate)
    return errorRate   #返回错误率，用于计算更加准确的错误率

'''
    n折交叉验证
    param: n折数
'''
def testNFoldSpam(n):
    totalError = 0
    for i in range(n):
        totalError += spamTest()
    print('sum=%f, average error is: %f' % (totalError, float(totalError/n)))

'''
    p69--RSS源分类器及高频词去除函数
    计算出现频率的函数
    params: vocabList: 词列表
        fullText：所有的文本
'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]

'''
    p69--RSS源分类器及高频词去除函数
    本地词汇
    这个方法与spamTest类似
    params: feed1:
        feed0:
'''
def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])  #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    #去掉出现次数最高的那些词开始
    top30Words = calcMostFreq(vocabList, fullText)
    print('top30Words:%s' % top30Words)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    #去掉出现次数最高的那些词结束

    trainingSet = list(range(2 * minLen))   #在python3中要，使用list，否则运行时会出错
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount)/len(testSet)
    print('the error rate is: ', errorRate)
    return vocabList, p0V, p1V, errorRate

'''
   p70--测试区域倾向的方法 
'''
def testNFoldTrend(n=1):
    #ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    ny = feedparser.parse('http://www.ifanr.com/feed')
    #sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://www.dushumashang.com/feed')

    print('ny:%s' % ny)
    print('sf:%s' % sf)
    totalErrorRate = 0
    for i in range(n):
        vocabList, pSF, pNY, errorRate = localWords(ny, sf)
        getTopWords(ny, sf)   #测试获取热门词的方法

        totalErrorRate += errorRate
    print('the average error rate is: %f' % (float(totalErrorRate)/n))

'''
    p71--最具表征性的词汇显示函数
    params: ny：参数1 
        sf：参数2
'''
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V, errorRate = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda  pair: pair[1], reverse = True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse = True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])


'''
    主函数区域
'''
if __name__ == "__main__":
    #testBayes()
    #testTrainNB0()
    #testingNB()   #调用朴素贝叶斯方法
    #spamTest()    #调用邮件检测方法
    #testNFoldSpam(10)   #调用N折邮件检测方法
    testNFoldTrend(10)

