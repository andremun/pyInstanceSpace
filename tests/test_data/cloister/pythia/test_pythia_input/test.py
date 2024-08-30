import numpy as np
import pandas as pd

CSV_Z = "z.csv"
z = pd.read_csv(CSV_Z,header=None,dtype=np.float64)
print(z.head())
from sklearn import preprocessing

# out_mu = np.mean(np.loadtxt('mu.csv',delimiter=','))#np.mean(z, axis=0)
# print("mu:")
# print(out_mu)
# # 计算矩阵每列的标准差
# out_sigma =np.std(np.loadtxt('sig.csv',delimiter=',')) #np.std(z, axis=0,ddof = 0)
# print("sigma:")
# print(out_sigma)

# 缩放数据，使每列标准差为1

# standardScaler = StandardScaler()
# scaler = StandardScaler()

# # 先使用 fit 方法
# scaler.fit(z)

# print(scaler)
# #(z - out_mu) / out_sigma
# standardScaler = StandardScaler().fit(z)
# mu = np.mean(z,axis=0)#dtype=np.float64)
# sigma = np.std(z,axis=0)#dtype=np.float64)
# data_transformed = (z-mu)/sigma#(z - np.mean(z))/np.std(z)
# def featureNormaliza(X):
#     X_norm = np.array(X)            #将X转化为numpy数组对象，才可以进行矩阵的运算
#     #定义所需变量
#     mu = 0 #np.zeros((1,X.shape[1]))
#     sigma =1 # np.zeros((1,X.shape[1]))

#     mu = np.mean(X_norm,0)          # 求每一列的平均值（0指定为列，1代表行）
#     sigma = np.std(X_norm,0)        # 求每一列的标准差
#     for i in range(X.shape[1]):     # 遍历列
#         X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]  # 归一化

#     return X_norm,mu,sigma
# data_transformed,mu,sig = featureNormaliza(z)
# data_transformed = preprocessing.scale(z)
data_transformed = (z-np.mean(z,axis=0))/np.std(z,ddof=1,axis=0)
pd.DataFrame(data_transformed).to_csv("z_qwq1.csv",header=None)
scaler = preprocessing.StandardScaler().fit(z)
# df = pd.DataFrame(data_transformed)
# znorm,mu,sig = zscore(z,axis=0,ddof=0)
# df = pd.DataFrame(znorm)

# df.to_csv('z_scaled.csv', header=False, index=False)
# df = pd.DataFrame(mu).to_csv('mu_output.csv',header=False,index=False)
# df = pd.DataFrame(sig).to_csv('sig_output.csv',header=False,index=False)

# print(mu)
# print(sig)
# X_scaled = preprocessing.scale(z)


