import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties #字体管理器  #参考文献：https://blog.csdn.net/asialee_bird/article/details/81027488

#设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 定义文本框和箭头格式
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

'''
    绘制带箭头的注释，实现课本p44效果
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, fontproperties = font, xy=parentPt, xycoords='axes fraction', \
                          xytext=centerPt, textcoords='axes fraction', \
                          va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

'''
    创建图形
'''
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

'''
    p49--3.8 使用决策树的分类函数
    params: inputTree:输入树
        featLabels:标签值
        testVec:测试向量
'''
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    createPlot()