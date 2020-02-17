
# coding: utf-8

# In[2]:

from math import log,exp,sqrt
from scipy import stats
from scipy.optimize import fsolve
import pandas as pd 
import numpy as np


# In[41]:

dataset_path = "D:\IV_index4.1.csv"
data_path=pd.read_csv(dataset_path,header=0)
data=data_path[:]
data.head(60)


# In[43]:

trading_dates=data.iloc[:,0].values
wind_code=data.iloc[:,1].values
strike=data.iloc[:,3].values
volatility_call=data.iloc[:,5].values
volatility_put=data.iloc[:,7].values
time=data.iloc[:,2].values/365
price=2.874
rate=data.iloc[:,8].values/100
call=data.iloc[:,4].values
put=data.iloc[:,6].values
len(call)
price


# In[44]:

#利用black-scholes实现期权定价
def blsprice(price,strike,rate,time,volatility):
   
    d1=(np.log(price/strike)+(rate+0.5*volatility**2)*time)/(volatility*np.sqrt(time))
    d2=d1-volatility*np.sqrt(time)
    call=price*stats.norm.cdf(d1,0.0,1.0)-strike*np.exp(-rate*time)*stats.norm.cdf(d2,0.0,1.0)
    put=strike*np.exp(-rate*time)*stats.norm.cdf(-d2,0.0,1.0)-price*stats.norm.cdf(-d1,0.0,1.0)
    return call


# In[45]:

call_BS=blsprice(price,strike,rate,time,volatility_call)
call_BS


# In[48]:

##反算隐含波动率
#volatility_est=np.ones(30)
#def impliedvolatility_call(call,price,strike,rate,time,volatility_est):
#    def difference(volatility_est,price,strike,rate,time):
#        est_call=blsprice(price,strike,rate,time,volatility_est)
#        return est_call-call
#    implied_vol=fsolve(difference,volatility_est,args=(price,strike,rate,time))
#    return implied_vol
def impliedvolatility_put(put,price,strike,rate,time,volatility_est):
    
    def difference(volatility_est,price,strike,rate,time):
        est_put=blsprice(price,strike,rate,time,volatility_est)
        return est_put-put
    implied_vol2=fsolve(difference,volatility_est,args=(price,strike,rate,time))
    return implied_vol2
#


# In[50]:

IV_call=impliedvolatility_call(put,price,strike,rate,time,volatility_est)
#IV_put=impliedvolatility_put(put,price,strike,rate,time,volatility_est)
total_date=data.iloc[:,2].values
IV_call


# In[45]:

dataframe=pd.DataFrame({"估计BS价格":call_BS,"估计delta":call_delta,"估计vega":vega2,"估计theta":theta_call,"估计gamma":gamma2,"估计rho":rho_call,"估计IV":IV_call,"期权执行价格":strike,"剩余自然日":total_date,"标的价格":price,"看跌期权价格":call,"无风险利率":rate,"编号":wind_code,"真实delta":delta,"真实vega":vega,"真实gamma":gamma,"真实rho":rho,"真实theta":theta,"真实IV":volatility})


# In[76]:

xp=pd.DataFrame([2.794,2.8])
fp=pd.DataFrame([0.379651,0.381079])
print(np.interp(2.85,xp,fp,right=0.4))


# In[3]:

df=pd.DataFrame({'strike':[2.794,2.8],'Call_IV':[0.379651,0.381079]})


# In[4]:

from sklearn import linear_model


# In[6]:

xp=pd.DataFrame(df['strike'])
xp


# In[9]:

y=pd.DataFrame(df['Call_IV'])
y


# In[10]:

regr=linear_model.LinearRegression()


# In[11]:

regr.fit(xp,y)
a,b=regr.coef_,regr.intercept_


# In[12]:

a


# In[13]:

b


# In[ ]:



