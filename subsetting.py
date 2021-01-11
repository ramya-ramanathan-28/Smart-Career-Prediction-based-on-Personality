
import pandas as pd
import numpy as np
import re

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('input/mbti_2.csv')
print(data.shape)
count=[0,0,0,0,0,0,0]
for index, row in data.iterrows():
        if row["type"]=="INTJ":
                print (index)
                #break
                count[0]=count[0]+1
                if count[0]>300:
                        data.drop(index, inplace=True)
        if row["type"]=="ENFP":
                print (index)
                #break
                count[1]=count[1]+1
                if count[1]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
        if row["type"]=="ENTJ":
                print (index)
                #break
                count[2]=count[2]+1
                if count[2]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
        if row["type"]=="INTP":
                print (index)
                #break
                count[3]=count[3]+1
                if count[3]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
        if row["type"]=="INFJ":
                print (index)
                #break
                count[4]=count[4]+1
                if count[4]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
        if row["type"]=="INFP":
                print (index)
                #break
                count[5]=count[5]+1
                if count[5]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
        if row["type"]=="ENTP":
                print (index)
                #break
                count[6]=count[6]+1
                if count[6]>300:
                        data.drop(index, inplace=True)
                        #df.drop(index, inplace=True)
print(data.shape)
data.to_csv('new_mbti.csv')
