
# coding: utf-8

# In[1334]:

from math import log,exp,sqrt
from scipy import stats
from scipy.optimize import fsolve
import pandas as pd 
import numpy as np
import math


# In[1335]:

#数据读取
dataset_path ="D:/IV_index1.29.csv"
data_path=pd.read_csv(dataset_path,header=0)
data=data_path[:]
data.head(10)


# In[1336]:

#赋值
k=data.iloc[:,3].values
t=data.iloc[:,2].values
R=3.5101/100
put=data.iloc[:,6].values
call=data.iloc[:,4].values


# In[1337]:

##到期时间大于7天的最小合约位置
delta_t=(t-7)
delta_t[delta_t<0]=365
np.where(delta_t==np.min(delta_t))


# In[1338]:

where=np.where(delta_t==np.min(delta_t))
pq=np.where(delta_t==np.min(delta_t))


# In[1339]:

T1=t[pq[0]]
T1=(T1[1]-1)/365

T1


# In[1340]:

kk=k[where]
n1=len(kk)
kk


# In[1341]:

#到期时间大于近月合约的最小合约位置
delta_t=(t-T1*365-1)
delta_t[delta_t<0.1]=365
np.where(delta_t==np.min(delta_t))


# In[1342]:

where2=np.where(delta_t==np.min(delta_t))


# In[1343]:

qqq=np.where(delta_t==np.min(delta_t))
T2=t[qqq[0]]
T2=(T2[0]-1)/365
T2


# In[1344]:

kk2=k[where2]
n2=len(kk2)
kk2


# In[1345]:

#计算F1的值
call2=call[where]
put2=put[where]
delta_s=abs(call2-put2)
local=np.where(delta_s==np.min(delta_s))
S=kk[local]
F1=S+exp(R*T1)*(call2[local]-put2[local])
#计算K01的值
b=F1-kk
b[b<0]=100
K01=F1-min(b)


# In[1346]:

F1


# In[1347]:

K01


# In[1348]:

#计算F2的值
call3=call[where2]
put3=put[where2]
delta_s=abs(call3-put3)
local=np.where(delta_s==np.min(delta_s))
S=kk2[local]
F2=S+exp(R*T2)*(call3[local]-put3[local])
#计算K02的值
b=F2-kk2
b[b<0]=100
K02=F2-min(b)


# In[1349]:

S


# In[1350]:

F2


# In[1351]:

K02


# In[1352]:

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
        V[n-1]=call2[n-1]*exp(R*T)*((kk[n-1]-kk[n-2])/(kk[n-1]**2))
        V[0]=put2[0]*exp(R*T)*((kk[1]-kk[0])/(kk[1]**2))
        IV=(2/T)*sum(V)
    IVX=IV-(1/T)*(((F/K0)-1)**2)
    return IVX


# In[1353]:

kk


# In[1354]:

call2


# In[1355]:

put2


# In[1356]:


#计算近月波动率和次近月波动率
Volatility_index1=IV_index(F1,kk,K01,R,T1,n1)
Volatility_index1


# In[1357]:

sigma1=sqrt(Volatility_index1)


# In[1358]:

sigma1


# In[1359]:

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


# In[1360]:

Volatility_index2=IV_index(F2,kk2,K02,R,T2,n2)
Volatility_index2


# In[1361]:

sigma2=sqrt(Volatility_index2)


# In[1362]:

sigma2


# In[1363]:

NT1=T1*365
NT1


# In[1364]:

NT2=T2*365
NT2


# In[1365]:

#计算IVX（有一部分近月与次近月波动率相同，选取近月波动率作为IVX）
a1=T1*(sigma1**2)*((NT2-30)/(NT2-NT1))
a2=T2*(sigma2**2)*(30-NT1)/(NT2-NT1)
ivx=100*sqrt((a1+a2)*365/30)
ivx


# In[1366]:

if T1*365>30:
    IVX2=sigma1*100
else:
    IVX2=ivx
IVX2


# In[1367]:

#打印数据
dataframe=pd.DataFrame({"估计IVX":IVX2,"F1值":F1,"K01":K01,"F2值":F2,"K02":K02,"估计sigma1":sigma1,"估计sigma2":sigma2,"T1":T1,"NT1":NT1,"T2":T2,"NT2":NT2,"真实IVX":18.4905})
dataframe


# In[1368]:

a=np.zeros(5)


# In[1369]:

a


# In[1370]:

for i in range (5):
    for j in range (5):
        a[i]+=i+j


# In[1371]:

a


# In[ ]:




# In[ ]:



