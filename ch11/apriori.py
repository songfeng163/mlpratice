'''
    ch11--使用Apriori算法来发现频繁集
'''

from numpy import *


'''
    p205--Apriori中的辅助方法
    加载数据的方法
'''
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

'''
    p205--Apriori中的辅助方法
    加载数据的方法
    param: dataSet：要转换的数据集
    return：转换后的1维数据
'''
# C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) #store all the item unrepeatly

    C1.sort()
    #return map(frozenset, C1)#frozen set, user can't change it.
    return list(map(frozenset, C1))  #python3中要转换成list数据

'''
    p205--Apriori中的辅助方法
    从C1中生成L1的函数
    params: D：数据集
        ck：侯选集合
        minSupport: 感兴趣集全的最小支持集
    return: retList: 返回的数据列表
        supportData: 支持数据
'''
def scanD(D,Ck,minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                #if not ssCnt.has_key(can):
                if not can in ssCnt:
                    ssCnt[can]=1
                else: ssCnt[can]+=1
    numItems=float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems #compute support
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

'''
    p206--测试上面的各个辅助函数的方法
'''
def testAuxiliary():
    dataSet = loadDataSet()
    print('dataSet如下：\n', dataSet)
    C1 = createC1(dataSet)
    print('C1如下：\n', C1)
    D = list(map(set, dataSet))   #python3中使用list进行转换
    print('D如下：\n', D)
    L1, suppData0 = scanD(D, C1, 0.5)   #0.5作为最小支持度
    print('L1如下：\n', L1)

'''
    p207--组织完整的Apriori算法
    生成所有的Apriori的函数
'''
#total apriori
def aprioriGen(Lk, k): #组合，向上合并
    #creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #两两组合遍历
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

'''
    p207--组织完整的Apriori算法
    生成单个的Apriori的函数
'''
#apriori
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet)) #python3
    L1, supportData = scanD(D, C1, minSupport)#单项最小支持度判断 0.5，生成L1
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):#创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = aprioriGen(L[k-2], k)#Ck
        Lk, supK = scanD(D, Ck, minSupport)  #get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1 #继续向上合并 生成项集个数更多的
    return L, supportData

'''
    p208--测试生成Apriori的方法
'''
def testAproiri():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    print('L如下:\n', L)
    print('suppData如下：\n', suppData)
    print('L[0]如下：\n', L[0])
    print('L[1]如下：\n', L[1])
    print('L[2]如下：\n', L[2])
    print('L[3]如下：\n', L[3])
    print('aprioriGen(L[0], 2)如下:\n', aprioriGen(L[0], 2))
    L, suppData = apriori(dataSet, minSupport=0.7)  #最小支持为0.7的结果
    print("L如下：\n", L)


'''
    p210--11.3关联规则生成函数
    params: L：频繁项集列表
        supprtData: 包含那些频繁项集支持数据的字典
        minConf：最小的可信度阈值
    return: 最大规则列表
'''
#生成关联规则
def generateRules(L, supportData, minConf=0.7):
    #频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = [] #存储所有的关联规则
    for i in range(1, len(L)):  #只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
            #如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)# 调用函数2
    return bigRuleList

'''
    p210--11.3关联规则生成函数
    params: freqSet: 频繁项集
        H: 可以出现在规则右部的元素列表H
        supportData:
        brl: 前面通过检查的bigRuleList
        minConf: 最小置信度
    return: prumedH：剪枝后的H
'''
#生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #针对项集中只有两个元素时，计算可信度
    prunedH = []#返回一个满足最小可信度要求的规则列表
    for conseq in H:#后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #可信度计算，结合支持度数据
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            #如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)#同样需要放入列表到后面检查
    return prunedH

'''
    p211--从最初的项集中生成更多的关联规则
    params: freqSet: 频繁项集
         H: 可以出现在规则右部的元素列表H
         supportData: 支持数据集
         brl: 前面通过检查的bigRuleList
         minConf：最小置信度
'''
#合并
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)#存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  #计算可信度
        if (len(Hmp1) > 1):    #满足最小可信度要求的规则列表多于1,则递归
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

'''
    p212--测试数据
'''
def testRulesApriori():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet, minSupport=0.5)
    rules = generateRules(L, suppData, minConf=0.7)  #最小置信度为0.7
    print('rules如下：\n', rules)
    rules = generateRules(L, suppData, minConf=0.5)  #最小置信度为0.5
    print('rules如下：\n', rules)


'''
    p213--投票功能的一些参数
'''
from time import sleep
from votesmart import votesmart

votesmart.apikey = '49024thereoncewasmanfromnantucketucket94040'  #这个apikey不能使用

'''
    p214--测试投票功能
'''
def testVoteBase():
    bills = votesmart.votes.getBillsByStateRecent()
    for bill in bills:
        print(bill.title, bill.billId)


def getActionIds():
    actionIDList = [];
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.vote.getBill(billNum)
            for action in billDetail.actions:
                if action.level == 'House' and \
                        (action.stage == 'Passage' or \
                         action.stage == 'Amendment Vote'):
                    actionID = int(action, actionID)
                    print('bill: %d has actionID :%d' % (billNum, actionID))
                    actionIDList.append(actionID)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)
    return actionIDList, billTitleList

'''
    内部测试方法
'''
if __name__ == '__main__':
    #pass
    #testAuxiliary()    #测试辅助函数的方法
    #testAproiri()    #测试Apriori方法
    #testRulesApriori()   #测试规则方法
    testVoteBase()     #投票的基本功能