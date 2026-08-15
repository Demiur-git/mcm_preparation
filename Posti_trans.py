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