import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#机器学习库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#统计分析库
import statsmodels.api as sm
from scipy import stats

#设置可视化风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

study_hours = np.random.uniform(0, 10, 100).reshape(-1, 1)

#生成模拟数据(附带随机噪声)
exam_score = 30 + 7 * study_hours + np.random.normal(0, 5, 100).reshape(-1, 1)

#转化为DataFrame,便于后续操作
data = pd.DataFrame({'study_hours':study_hours.flatten(), 'exam_score':exam_score.flatten()})

#绘制散点图
plt.figure(figsize=(10,5))
sns.scatterplot(x='study_hours', y='exam_score', data=data)
plt.title('学习时间与考试成绩的关系')
plt.xlabel('学习时间(小时)')
plt.ylabel('考试成绩')
plt.show()
plt.close()

#计算相关系数
correlation = data['study_hours'].corr(data['exam_score'])
print(f"学习时间与考试成绩的相关系数为: {correlation:.4f}")

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[['study_hours']], data['exam_score'], test_size=0.2, random_state=42)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

#训练模型
model = LinearRegression()
model.fit(X_train, y_train)

#获取模型参数
intercept = model.intercept_  # 截距
slope = model.coef_[0]  # 斜率
print(f"斜率: {slope:.4f}")
print(f"截距: {intercept:.4f}")

print("回归方程为：")
print(f"exam_score = {slope:.4f} * study_hours + {intercept:.4f}")

#预测测试集
y_pred = model.predict(X_test)
print(f"预测值: {y_pred}")

predict_df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(predict_df.head())

#计算各种评估指标
r2 = r2_score(y_test, y_pred)
print(f"R^2: {r2:.4f}")
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差(MSE): {mse:.4f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"平均绝对误差(MAE): {mae:.4f}")

#结果可视化
plt.figure(figsize=(10,5))

#原始成绩散点图
plt.scatter(X_test, y_test, label='Actual', color='blue', alpha=0.7)

#添加回归线
plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth=2)
plt.legend()
plt.title('学习时间与考试成绩的关系')
plt.xlabel('学习时间(小时)')
plt.ylabel('考试成绩')
plt.show()
plt.close()

#绘制残差图
plt.figure(figsize=(10,5))
sns.scatterplot(x=y_pred, y=y_test - y_pred, color='blue', alpha=0.7)
plt.title('残差图')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.show()
plt.close()

#进行预测
test = np.array([[8]])
print(f"预测值: {model.predict(test)[0]:.2f}")

#封装函数
def run_linear_regression_analysis(data, feature_col, target_col):
    """
    执行一元线性回归分析的完整流程模板。

    参数:
    data (pd.DataFrame): 包含数据的DataFrame。
    feature_col (str): 自变量列名。
    target_col (str): 因变量列名。

    返回:
    dict: 包含模型、评估指标和预测结果的字典。
    """
    # 1. 数据准备和划分
    X = data[[feature_col]]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 模型训练
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. 预测与评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 4. 打印结果
    print(f'--- 分析报告: {feature_col} vs {target_col} ---')
    print(f'回归方程: {target_col} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {feature_col}')
    print(f'R-squared (R²): {r2:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    # 5. 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='实际值')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测回归线')
    plt.title(f'{feature_col} 与 {target_col} 的回归分析')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'predictions_df': pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    }


# 使用模板函数
results = run_linear_regression_analysis(data, 'study_hours', 'exam_score')
