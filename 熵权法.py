import numpy as np
import math

#将原始数据转化为极大型数据
def primi_to_max(A):
    A_copy = A
    max_a = max(A)
    min_a = min(A)
    A_copy = (A_copy - min_a) / (max_a - min_a)
    return A_copy

#将极小型数据转化为极大型数据
def min_to_max(A):
    A_copy = A
    max_a = max(A)
    min_a = min(A)
    A_copy = (max_a - A_copy) / (max_a - min_a)
    return A_copy

#将中间型数据转化为极大型数据
def mid_to_max(A,best_a):
    A_copy = A
    M = max(abs(A_copy - best_a))
    A_copy = 1 - abs(A_copy - best_a)/M
    return A_copy

def range_to_max(A,best_a,best_b):
    A_copy = A
    M = max(best_a - min(A_copy), max(A_copy) - best_b)
    for i in range(len(A_copy)):
        if A_copy[i] < best_a:
            A_copy[i] = 1 - (best_a - A_copy[i])/M
        elif A_copy[i] > best_b:
            A_copy[i] = 1 - (A_copy[i] - best_b)/M
        else:
            A_copy[i] = 1
    return A_copy

#对数化矩阵（如果P内元素为0则直接取0）
def log_P(P,n,m):
    P_copy = P.copy()
    for i in range(n):
        for j in range(m):
            if P_copy[i,j] == 0:
                P_copy[i,j] = 0
            else:
                P_copy[i,j] = np.log(P_copy[i,j])
    return P_copy

#用户输入参评数量与指标数量
n = int(input("请输入参评数量："))
m = int(input("请输入指标数量："))

#向量类型输入
print("请输入类型矩阵：1:极大型，2：极小型，3：中间型，4：区间型")
kind = input().split()

#接收用户输入的矩阵
print("请输入矩阵：")
A = np.zeros((n,m))
for i in range(n):
    A[i] = input().split() #接收用户输入的每一行数据
    A[i] = list(map(float,A[i])) #将字符串转换为浮点数
print(f"输入矩阵A为：\n{A}")

#将矩阵转化为极大型矩阵（原始矩阵正向化）
for i in range(m):
    if kind[i] == "2":
        A[:,i] = min_to_max(A[:,i])
    elif kind[i] == "3":
        best_a = float(input("请输入中间型指标的最优值："))
        A[:,i] = mid_to_max(A[:,i],best_a)
    elif kind[i] == "4":
        best_a = float(input("请输入区间型指标的最优最小值a："))
        best_b = float(input("请输入区间型指标的最优最大值b："))
        A[:,i] = range_to_max(A[:,i],best_a,best_b)
    else:
        A[:,i] = primi_to_max(A[:,i])
print(f"正向化矩阵A为：\n{A}")

#正向化矩阵标准化
#确保A为浮点数矩阵
A = A.astype(float)

#标准化操作
power_A = np.power(A,2)
sum_power_A = np.sum(power_A,axis=0)
stand_A = A/np.sqrt(sum_power_A)
print(f"标准化矩阵A为：\n{stand_A}")

#归一化操作
sum_stand_A = np.sum(stand_A,axis=0)
P = stand_A/sum_stand_A
print(f"归一化矩阵A为：\n{P}")

#计算信息熵值
log_P = log_P(P,n,m)
E = -np.sum(P*log_P,axis=0)/math.log(n)
print(f"信息熵值为：{E}")

#计算信息效用值和权重
D = 1 - E
print(f"信息效用值为：{D}")
weight = D/np.sum(D)
print(f"权重为：{weight}")