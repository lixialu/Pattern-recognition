import csv
import re
from numpy import *
# 读取文件
def read_data(file):
    train_data = csv.reader(open(file, encoding="utf-8"))
    lines = 0
    for r in train_data:
        lines += 1
    train_data_label = zeros([lines - 1, ])
    train_data_content = []
    train_data = csv.reader(open(file, encoding="utf-8"))
    i = 0
    for data in train_data:
        if data[0] == "Label" or data[0] == "SmsId":
            continue
        if data[0] == "ham":
            train_data_label[i] = 0
        if data[0] == "spam":
            train_data_label[i] = 1
        train_data_content.append(data[1])
        i += 1
    return train_data_label, train_data_content

    # 简化文本
def textsimpfy(string):
    listwords = re.split(r" |\!|\?|\.|\,|\t|\;|\*|\'|\"|\-", string)
    return [word.lower() for word in listwords if len(word) > 2]  # 只取具有价值的单词，所以长度大于2


# 创建所有单词的单词表
def Creatvocablist(datalist):
    vocabset = set([])
    for x in datalist:
        vocabset = vocabset | set(x)
    return list(vocabset)


# 创建[0,1]向量
def vector(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        try:
            returnvec[vocablist.index(word)] += 1
        except:
            pass
    return returnvec


def creatmat(data_content, vocabulist):
    train_content = []
    for data in data_content:
        train_content.append(vector(vocabulist, data))
    return train_content


# 训练函数
def trainNB(trainmat, trainlabel):
    numTraindocs = len(trainmat)
    numwords = len(trainmat[0])
    pre1 = sum(trainlabel) / float(numTraindocs)  # 类别为垃圾邮件的先验概率
    p0 = ones(numwords)  # 初始化正常邮件对应的各个特征数量
    p1 = ones(numwords)  # 初始化垃圾邮件对应的各个特征数量
    sump0 = 2.0  # 正常邮件的词数
    sump1 = 2.0  # 垃圾邮件的词数
    for i in range(numTraindocs):
        if trainlabel[i] == 0:
            p0 += trainmat[i, :]
            sump0 += sum(trainmat[i, :])
        else:
            p1 += trainmat[i, :]
            sump1 += sum(trainmat[i, :])
    p0vect = log(p0 / sump0)  # 列表里每一项为各个词（特征）对应的条件概率（正常邮件）
    p1vect = log(p1 / sump1)
    return p0vect, p1vect, pre1


def classfyNB(vec, p0vect, p1vect, pre1):
    post0 = sum(vec * p0vect) + log(1 - pre1)  # 属于正常邮件的后验概率
    post1 = sum(vec * p1vect) + log(pre1)  # 属于垃圾邮件的后验概率
    if post1 > post0:
        return 1
    else:
        return 0

#logestic回归
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 随机梯度下降算法
def stocGradAscent(dataMat,labelMat):
     m,n=shape(dataMat)
     #print(n)
     aplha=0.01
     weights=ones(n)
    # print(weights)
     for i in range(m):
        h=sigmoid(sum(dataMat[i]*weights))
        error=labelMat[i]-h
        weights=weights-aplha*error*dataMat[i]
     return weights

def classifyVector(inX,weights):
     weights=mat(weights)
     prob=sigmoid(sum(weights*inX))
     if prob>0.5:
        return 1
     else:
        return 0


# 测试
train_data_label, train_data_content = read_data("C:/train.csv")
test_data_label, test_data_content = read_data("C:/test.csv")
train_txt = []
for i in range(len(train_data_content)):
    train_txt.append(textsimpfy(train_data_content[i]))
    i += 1
text_txt = []
for i in range(len(test_data_content)):
    text_txt.append(textsimpfy(test_data_content[i]))
print(text_txt )
vocabulist = Creatvocablist(train_txt)
train_content = array(creatmat(train_txt, vocabulist))
test_content = creatmat(text_txt, vocabulist)
p0, p1, pre1 = trainNB(train_content, array(train_data_label))

rightcount = 0
for i in range(len(test_content)):
    if classfyNB(array(test_content[i]), p0, p1, pre1) == test_data_label[i]:
        rightcount += 1
print(rightcount / len(test_content))

trainWeights=stocGradAscent(array(train_content),train_data_label)
test_label=[]
test_data= csv.reader(open("C:/test.csv", encoding="utf-8"))

for data in test_data:
    if data[0]=="SmsId":
        continue
    test_label.append(int(data[0]))

testweights=[]
for i in test_label:
    testweights.append(trainWeights[i])
errorCount=0
if classifyVector(array(test_content),testweights)!=train_data_label[i]:
  errorCount+=1
errorRate=(errorCount/len(test_content))
print(1-errorRate)