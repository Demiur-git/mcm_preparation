import numpy as np
from Posti_trans import min_to_max,mid_to_max,range_to_max

#统一量纲变化函数定义
def mean_trans(A):
    A_copy = A
    A_mean = np.mean(A_copy,axis=0)
    A_copy = A_copy / A_mean
    return A_copy

def original_trans(A):
    A_copy = A
    temp = []
    for i in range(len(A_copy[0])):
        for j in range(len(A_copy)):
            if A_copy[j,i] != 0:
                temp.append(A_copy[j,i])
                break
            else:
                pass
    A_l = np.array(temp)
    A_copy = A_copy / A_l
    return A_copy

def percent_trans(A):
    A_copy = A
    max_A = np.max(A_copy,axis=0)
    A_copy = A_copy / max_A
    return A_copy

def multi_trans(A):
    A_copy = A
    temp = []
    for i in range(len(A_copy[0])):
        if np.min(A_copy[:,i]) != 0:
            temp.append(np.min(A_copy[:,i]))
        else:
            temp_min = A_copy[0, i]
            for j in range(len(A_copy)):
                if (A_copy[j,i] != 0 and A_copy[j,i] < temp_min) or (temp_min==0):
                    temp_min = A_copy[j,i]
            temp.append(temp_min)
    A_copy = A_copy / np.array(temp)
    return A_copy

def nol_trans(A):
    A_copy = A
    A_copy = A_copy / A_copy[0,:]
    return A_copy

def max_trans(A):
    A_copy = A
    A_min = np.min(A_copy)
    A_max = np.max(A_copy)
    A_copy = A_copy - A_min / A_max
    return A_copy

def min_trans(A):
    A_copy = A
    A_min = np.min(A_copy)
    A_max = np.max(A_copy)
    A_copy = (A_max - A_copy) / (A_max - A_min)
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

#根据用户输入选择统一量纲变化函数
print("请输入统一量纲变化函数：1：均值化，2：初始值，3：百分比，4：倍数，5：归一化，6：最大法化，7：区间值化")
func = input()

if func == "1":
    X_f = mean_trans(A)
elif func == "2":
    X_f = original_trans(A)
elif func == "3":
    X_f = percent_trans(A)
elif func == "4":
    X_f = multi_trans(A)
elif func == "5":
    X_f = nol_trans(A)
elif func == "6":
    X_f = max_trans(A)
elif func == "7":
    X_f = min_trans(A)
else:
    print("输入错误")
    exit()
print(f"统一量纲变化后的矩阵X_f为：\n{X_f}")

#构造虚拟母序列（用每一列的最大值来构成母序列）
X_f_max = np.max(X_f,axis=0)

#计算子序列与母序列差值
X_f_diff = abs(X_f - X_f_max)
print(f"子序列与母序列差值为：\n{X_f_diff}")

#计算关联系数矩阵
p = float(input("请输入分辨系数ρ："))
a = np.min(X_f_diff)
b = np.max(X_f_diff)
Xi = (a + p*b) / (X_f_diff + p*b)
print(f"关联系数矩阵Xi为：\n{Xi}")

#计算关联度
R = np.mean(Xi,axis=1)
print(f"关联度R为：\n{R}")
