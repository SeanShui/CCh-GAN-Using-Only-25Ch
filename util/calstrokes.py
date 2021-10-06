#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt 

#  這個選字程式使用 stroke embedding dataset from https://github.com/JinshanZeng/Stroke_Based_Chinese_Character_Generation_Dataset


# 計算全部字碼筆劃分佈和每個字碼包括的筆劃種類數
i = 0
embeddingCnt = []
strokeCntA = [0] * 32
strokeCntB = [0] * 32
maxEmbeddingCnt = 0
minEmbeddingCnt = 100

fp = open('Regular Script.txt', "r")
line = fp.readline() 
while i<2500:  
    line = fp.readline()
    embedding = line.split()[1:]
    Cnt = 0
    print(embedding)
    
    j = 0
    while j<32:
       if embedding[j]=='1':
          Cnt = Cnt + 1
       j = j + 1

    embeddingCnt.append(Cnt)
    if Cnt>maxEmbeddingCnt:
       maxEmbeddingCnt = Cnt
    if Cnt<minEmbeddingCnt:
       minEmbeddingCnt = Cnt

    print('stroke embedding={}'.format(embedding))
    i = i  + 1

fp.close()

print('\n')    
print('embeddingCnt={}'.format(embeddingCnt))
print('\n')
print('maxEmbeddingCnt={}'.format(maxEmbeddingCnt))
print('minEmbeddingCnt={}'.format(minEmbeddingCnt))

n, bins, patches = plt.hist(embeddingCnt)
plt.show()

# 用每個字筆劃種類數=5 當做 threshold 選出 A B set,確保 A B set 每種筆劃都至少出現一次
setA = []
setB = []
i=0

fp = open('Regular Script.txt', "r")
line = fp.readline()
while i<2500:  
    line = fp.readline()
    embedding = line.split()[1:]
    fname = line.split()[0]
    
    if embeddingCnt[i]>5:
       j = 0
       choose = 0
       for j in range(32):
          if embedding[j]=='1' and strokeCntA[j]==0:
             strokeCntA[j] = strokeCntA[j] + 1
             choose = 1
             break
       if choose==1:
             setA.append(fname)
    else:
       j = 0
       choose = 0
       for j in range(32):
          if embedding[j]=='1' and strokeCntB[j]==0:
             strokeCntB[j] = strokeCntB[j] + 1
             choose = 1
             break
       if choose==1:
             setB.append(fname)

    i = i  + 1

fp.close()

print('strokeCntA={}'.format(strokeCntA))
print('strokeCntB={}'.format(strokeCntB))
print('setA={}'.format(setA))
print('setA number={}'.format(len(setA)))
print('setB={}'.format(setB))
print('setB number={}'.format(len(setB)))

