import numpy as np

#将极小型数据转化为极大型数据
def min_to_max(A):
    A_copy = A
    max_a = max(A_copy)
    A_copy = max_a - A_copy
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
        pass
print(f"正向化矩阵A为：\n{A}")

#正向化矩阵标准化
#确保A为浮点数矩阵
A = A.astype(float)

#标准化操作
power_A = np.power(A,2)
sum_power_A = np.sum(power_A,axis=0)
stand_A = A/np.sqrt(sum_power_A)
print(f"标准化矩阵A为：\n{stand_A}")

#接收用户输入的权重
print("请输入各指标的权重，用空格分隔，总和应为1：")
w = list(map(float,input().split()))
print(f"权重为：{w}")

#寻找理想最优解与最劣解
stand_A_max = np.max(stand_A,axis=0)
stand_A_min = np.min(stand_A,axis=0)
print(f"理想最劣解为：{stand_A_min}")
print(f"理想最优解为：{stand_A_max}")

#计算Topsis值
w = np.array(w)
d_plus = np.sqrt(np.sum(w*np.square(stand_A_max - stand_A),axis=1))
d_minus = np.sqrt(np.sum(w*np.square(stand_A_min - stand_A),axis=1))
print(f"d_plus为：{d_plus}")
print(f"d_minus为：{d_minus}")
s = d_minus/(d_plus + d_minus)
print(f"Topsis值为：{s}")