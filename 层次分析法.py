import numpy as np

#一致性检验
A = np.array([[1,5,3,2],[1/5,1,1/2,1/3],[1/3,2,1,1/2],[1/2,3,2,1]])

n = A.shape[0]

eig_val,eig_vec = np.linalg.eig(A)
max_eig = max(eig_val)

CI = (max_eig - n)/(n-1)
CR = [0,0.0001,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54,1.56,1.58,1.59]

CR = CI/CR[n]

print(f"一致性检验CI={CI}")
print(f"一致性检验CR={CR}")

if CR < 0.10:
    print("因为CR<0.1，所以A的一致性检验可以接受")
else:
    print("A的一致性检验不可接受，需要重新调整")

#算数平均法计算权重
#按列相加
ASum = np.sum(A,axis=0)

#归一化
stand_A = A/ASum

#计算权重
ASum_w = np.sum(stand_A,axis=1)

weight = ASum_w/n

print(weight)

#几何平均法计算权重
#按行相乘
prod_A = np.prod(A,axis=1)

power_A = np.power(prod_A,1/n)

sum_power_A = np.sum(power_A)

weight = power_A/sum_power_A

print(weight)

#特征值法计算权重
#找到最大特征值对应的特征向量进行归一化
max_index = np.argmax(eig_val)
max_vec = eig_vec[:,max_index]

weight = max_vec/np.sum(max_vec)

print(weight)