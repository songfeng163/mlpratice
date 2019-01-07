'''
    p177起的--tkinter的使用测试
'''
from numpy import *
from tkinter import *   #注意python3中为小写的tkinter
import regTrees   #导入自定义的回归树模块
#绘图与tk的结合使用
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

'''
    p177的简单的测试程序
'''
def testSimpleTk():
    root = Tk()
    myLabel = Label(root, text='Hello World')
    myLabel.grid()  #网格布局管理器
    root.mainloop()



'''
    p178--9.6 重绘方法
'''
def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a =reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        #print('模型树')
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        #print('普通树')
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    print('myTree: \n%s' % myTree)
    print('\nyHat: \n%s' % yHat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)  #这里使用tolist方法，否则出现1-D的异常
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()  #20190107发现新版的python中不使用这个show方法，使用的话，会出错，通过在jyputer中测试发现，这句话不能注释，注释后，图形不能更新

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('enter Integer for tolN')
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print('enter Float for tolS')
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

'''
    p178--9.6绘制树的方法
'''
def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

'''
    p178--9.6 测试绘制树的方法
'''
#def testDrawTree():
root = Tk()
#Label(root, text='Plot place Holder').grid(row=0, columnspan=3)
#p179的代码替换上面的一行代码开始,显示绘图区域
reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
#reDraw.canvas.show()  #20190107测试发现，在新的版本中这句话用不到了，用到的话，会出现异常:FigureCanvasTkAgg没有方法show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# p179的代码替换上面的一行代码结束,显示绘图区域

Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')

Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

Button(root, text='Quit', command=root.quit).grid(row=1, column=2)

Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=3)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()



'''
    内部测试方法
'''
if __name__ == '__main__':
    #testSimpleTk()    #测试简单的图形
    #testDrawTree()     #绘制树的测试
    pass
