
# coding: utf-8

# In[37]:

from math import log,exp,sqrt
from scipy import stats
from scipy.optimize import fsolve
import pandas as pd 
import numpy as np
import math


# In[52]:

#数据读取
dataset_path ="D:/total_VIX.csv"
data_path=pd.read_csv(dataset_path,header=0)
DATA=data_path[:]
DATA1=DATA.set_index('trading_date')
data=DATA1.ix['2019/1/2']
data.head(2)


# In[53]:

#赋值
k=data.iloc[:,2].values
t=data.iloc[:,1].values
put=data.iloc[:,5].values
call=data.iloc[:,3].values
Rf=data.iloc[:,7].values
rf=Rf[1]/100


# In[54]:

##到期时间大于7天的最小合约位置
delta_t=(t-7)
delta_t[delta_t<0]=365
np.where(delta_t==np.min(delta_t))
where=np.where(delta_t==np.min(delta_t))
pq=np.where(delta_t==np.min(delta_t))
T1=t[pq[0]]
T1=(T1[1]-1)/365
kk=k[where]
n1=len(kk)


# In[55]:

#到期时间大于近月合约的最小合约位置
delta_t=(t-T1*365-1)
delta_t[delta_t<0.1]=365
np.where(delta_t==np.min(delta_t))
where2=np.where(delta_t==np.min(delta_t))
pq=np.where(delta_t==np.min(delta_t))
T2=t[pq[0]]
T2=(T2[0]-1)/365
kk2=k[where2]
n2=len(kk2)


# In[56]:

#计算F1的值
call2=call[where]
put2=put[where]
delta_s=abs(call2-put2)
local=np.where(delta_s==np.min(delta_s))
S=kk[local]
F1=S+exp(rf*T1)*(call2[local]-put2[local])
#计算K01的值
b=F1-kk
b[b<0]=100
K01=F1-min(b)
#判断合约是否需要补充
if n1-local[0][0]<4:
    vv=4-n1+local[0][0]
    print("近月单边call合约少于4个需补充：",vv)
elif local[0][0]<4:
    ff=4-local[0][0]
    print("近月单边put合约少于4个需补充",ff)
else:
    print("近月双边合约数正常")


# In[57]:

#计算F2的值
call3=call[where2]
put3=put[where2]
delta_s=abs(call3-put3)
local=np.where(delta_s==np.min(delta_s))
S=kk2[local]
S
F2=S+exp(rf*T2)*(call3[local]-put3[local])
#计算K02的值
b=F2[0]-kk2
b[b<0]=100
K02=F2[0]-min(b)
#判断合约是否需要补充
if n2-local[0][0]<4:
    vv=4-n2+local[0][0]
    print("次近月单边call合约少于4个需补充：",vv)
elif local[0][0]<4:
    ff=4-local[0][0]
    print("次近月单边put合约少于4个需补充",ff)
else:
    print("近月双边合约数正常")


# In[58]:

#构造近月波动率函数
def IV_index(F,kk,K0,R,T,n):
    IV=0
    P=np.zeros(n)
    delta_k=np.zeros(n)
    V=np.zeros(n)
    for i in range(1,n-1):
        if kk[i]<K0:
            P[i]=put2[i]
            delta_k[i]=(kk[i+1]-kk[i-1])/2
            V[i]=(delta_k[i]/(kk[i]**2))*exp(R*T)*P[i]
        elif kk[i]>K0:
            P[i]=call2[i]
            delta_k[i]=(kk[i+1]-kk[i-1])/2
            V[i]=(delta_k[i]/(kk[i]**2))*exp(R*T)*P[i]
        else: 
            P[i]=(call2[i]+put2[i])/2
            delta_k[i]=(kk[i+1]-kk[i-1])/2
            V[i]=(delta_k[i]/(kk[i]**2))*exp(R*T)*P[i]
       #V[n-5]=call2[n-5]*exp(R*T)*((kk[n-5]-kk[n-6])/(kk[n-5]**2))
        #V[n-5]=0
        #V[n-4]=0
        #V[n-3]=0
        #V[n-2]=0
        #V[n-1]=0
        #V[n-6]=call2[n-6]*exp(R*T)*((kk[n-6]-kk[n-7])/(kk[n-6]**2))
        V[n-1]=call2[n-1]*exp(R*T)*((kk[n-1]-kk[n-2])/kk[n-1]**2)
        V[0]=put2[0]*exp(R*T)*((kk[1]-kk[0])/(kk[1]**2))
        IV=(2/T)*sum(V)
    IVX=IV-(1/T)*(((F/K0)-1)**2)
    return IVX


# In[59]:

#计算近月波动率
Volatility_index1=IV_index(F1,kk,K01,rf,T1,n1)
sigma1=sqrt(Volatility_index1)
sigma1


# In[60]:

#构造次近月波动率函数
def IV_index(F,kk,K0,R,T,n):
    IV=0
    P=np.zeros(n)
    delta_k=np.zeros(n)
    V=np.zeros(n)
    for i in range(1,n-1):
        if kk2[i]<K0:
            P[i]=put3[i]
            delta_k[i]=(kk2[i+1]-kk2[i-1])/2
            V[i]=(delta_k[i]/(kk2[i]**2))*exp(R*T)*P[i]
        elif kk2[i]>K0:
            P[i]=call3[i]
            delta_k[i]=(kk2[i+1]-kk2[i-1])/2
            V[i]=(delta_k[i]/(kk2[i]**2))*exp(R*T)*P[i]
        else: 
            P[i]=(call3[i]+put3[i])/2
            delta_k[i]=(kk2[i+1]-kk2[i-1])/2
            V[i]=(delta_k[i]/(kk2[i]**2))*exp(R*T)*P[i]
        V[n-1]=call3[n-1]*exp(R*T)*((kk2[n-1]-kk2[n-2])/kk2[n-1]**2)
        V[0]=put3[0]*exp(R*T)*((kk2[1]-kk2[0])/kk2[1]**2)
        IV=(2/T)*sum(V)
    IVX=IV-(1/T)*(((F/K0)-1)**2)
    return IVX


# In[61]:

#计算次近月波动率
Volatility_index2=IV_index(F2,kk2,K02,rf,T2,n2)
sigma2=sqrt(Volatility_index2)
sigma2


# In[62]:

#计算IVX（有一部分近月与次近月波动率相同，选取近月波动率作为IVX）
NT1=T1*365
NT2=T2*365
a1=T1*(sigma1**2)*((NT2-30)/(NT2-NT1))
a2=T2*(sigma2**2)*((30-NT1))/(NT2-NT1)
ivx=100*sqrt((a1+a2)*365/30)


# In[63]:

#判断近月合约是否大于30天
if T1*365>30:
    IVX2=sigma1*100
else:
    IVX2=ivx
IVX2


# In[64]:

#打印数据
dataframe=pd.DataFrame({"估计IVX":IVX2,"F1值":F1,"K01":K01,"F2值":F2,"K02":K02,"估计sigma1":sigma1,"估计sigma2":sigma2,"T1":T1,"NT1":NT1,"T2":T2,"NT2":NT2,"rf":rf})
dataframe

