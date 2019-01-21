'''
    20190121 ch12 使用FP-Growth算法来高效发现频繁项集
'''
from numpy import *
import twitter   #导入twitter库
from time import sleep
import re

'''
    p226--12.1 FP树的定义
    节点类的定义
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None   #用于链接相似的元素项
        self.parent = parentNode  #父节点
        self.children = {}   #子节点

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

'''
    p226--测试节点创建的方法
'''
def testNode():
    rootNode = treeNode('pyramid', 9, None)   #节点'金子塔'
    rootNode.children['eye'] = treeNode('eye', 13, None)
    #rootNode.disp()
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)   #节点'凤凰'
    rootNode.disp()

'''
    p228--12.2 FP树构建函数
    创建树的方法
    params: dataSet：数据集
        minSup: 最小支持度的值
    return:
        retTree: 生成的树
        headerTable：表的头指针
'''
def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):   #1.循环移除不满足最小支持度的元素项
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:   #2.如果没有元素滞要求，则退出
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:   #3.根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)   #4.使用排序后的频率项集对树进行填充
    return retTree, headerTable

'''
    p228--12.2 FP树构建函数
    更新树的方法
'''
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:    #5.对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

'''
    p228--12.2 FP树构建函数
    更新头指针的方法
'''
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

'''
    p230--12.3简单数据集及数据包装器
    简单的数据集
'''
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

'''
    p230--12.3简单数据集及数据包装器
    创建初始化数据集函数
'''
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

'''
    p230--测试生成数据的方法
'''
def testCreateDataSet():
    simData = loadSimpDat()
    print('simData数据如下：\n', simData)
    initSet = createInitSet(simData)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()   #显示数据

'''
    p232--12.4发现以给定元素项结尾的所有路径的函数
    对树进行上溯
    params: leafNode: 叶子节点
        prefixPath: 前置路径
    return: 整个路径
'''
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:    #迭代上溯整棵树，并收集所有遇到的元素项的名称
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

'''
    p232--12.4发现以给定元素项结尾的所有路径的函数
    查找前置路径，遍历链表直到结尾
    params: basePat：模式基
        treeNode：节点
    return:condPats: 条件模式字典
'''
def findPrefixPath(basePat, treeNode):
    condPats = {}   #条件模式基字典
    while treeNode != None:   #对于每个节点，都使用ascendTree上溯到FP树
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

'''
    p232--测试查找条件模式基的函数
    在上一个测试testCreateDataSet基础上进行更新
'''
def testFindPrefixPath():
    simData = loadSimpDat()
    print('simData数据如下：\n', simData)
    initSet = createInitSet(simData)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    #myFPtree.disp()  # 显示数据
    condPats = findPrefixPath('x', myHeaderTab['x'][1])
    print('x的condPats的内容如下：\n', condPats)
    condPats = findPrefixPath('r', myHeaderTab['r'][1])
    print('r的condPats的内容如下：\n', condPats)
    condPats = findPrefixPath('z', myHeaderTab['z'][1])
    print('z的condPats的内容如下：\n', condPats)

'''
    p233--12.5递归查找频繁项集的mineTree函数
'''
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:str(p[1]))]   #从头指针表的底端开始
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)   #从条件模式基来构建条件FP树

        if myHead != None:
            print('conditional tree for: ', newFreqSet)  # 用于p234显示数据而添加的代码
            myCondTree.disp(1)  # 用于p234显示数据而添加的代码

            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  #挖掘条件FP树

'''
    p234--条件树的测试方法
    在上一个testFindPrefixPath函数的基础上进行更新
'''
def testMineTree():
    simData = loadSimpDat()
    print('simData数据如下：\n', simData)
    initSet = createInitSet(simData)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    freqItems = []   #建立一个空的频繁项集
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)

'''
    p236--12.6 访问twitter Ptyhon库的代码
    因为无法登录twitter网站，也没有办法申请到api账号，故未完成本功能以及之后的功能
'''
def getLotsOfTweets(searchStr):
    CONSUMER_KEY = 'get when you create an app'
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print('feteching page %d' % i)
        searchResults = api.getSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

'''
    p236--测试获取twitter数据的方法
'''
def testGetTweets():
    lotsOfTweets = getLotsOfTweets('RIMM')
    print('lotsOfTweets[0][4]的内容：\n', lotsOfTweets[0][4].text)

'''
    内部测试方法
'''
if __name__ == '__main__':
    #pass   #点位符
    #testNode()    #测试节点函数的方法
    #testCreateDataSet()      #测试创建数据集的方法
    #testFindPrefixPath()     #测试查找条件模式基
    testMineTree()      #测试条件树的方法