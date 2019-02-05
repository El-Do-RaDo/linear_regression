import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading the dataset and making the list
ds=pd.read_csv("/home/spider/WORK/DATA/train.csv")
ts=pd.read_csv("/home/spider/WORK/DATA/test.csv")
ds[np.isnan(ds)]=0
x_list=list(ds['x'][:100])
y_list=list(ds['y'][:100])
x=np.array(x_list)
y=np.array(y_list)
plt.scatter(x_list,y_list,color="c")
n=np.size(x)
#plt.show()

#for training purpose
mx=np.mean(x)
my=np.mean(y)
sxy=((np.sum(y*x))-(n*mx*my))
sxx=((np.sum(x*x))-(n*mx*mx))

#for finding the intercept and slope 
m=(sxy/sxx)
c=(my-m*mx)

#prediction of the values for the x
xp=np.array(ts['x'])
con=np.size(xp)
y_pre=[]
for i in range(con):
	temp=(c+(m*xp[i]))
	y_pre.append(temp)
y_a=np.array(y_pre)

#for making the linear regression line
plt.plot(xp,y_a,color="m")
plt.xlabel('x labels')
plt.ylabel('y labels')
plt.show()
