# 文件路径: arima_sales_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 1. 生成模拟的2022-2023年的月销量数据
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
sales_data = np.random.randint(100, 500, size=len(dates))

# 创建DataFrame
df = pd.DataFrame({'Date': dates, 'Sales': sales_data})
df.set_index('Date', inplace=True)

# 打印生成的模拟数据
print("生成的销量数据：")
print(df)

# 2. 数据可视化
plt.figure(figsize=(10, 5))
plt.plot(df, marker='o')
plt.title('Monthly Sales Data (2022-2023)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# 3. ARIMA 模型训练
# 拆分训练集和测试集 (2022用于训练，2023用于测试)
train_data = df[df.index < '2023-07-01']
test_data = df[df.index >= '2023-07-01']

# 创建并拟合ARIMA模型 (order=(1,1,1) 是常用的初始参数)
model = ARIMA(train_data, order=(1, 1, 1))
arima_result = model.fit()

# 打印模型摘要
print(arima_result.summary())

# 4. 在测试集上进行预测
forecast = arima_result.forecast(steps=len(test_data))
test_data['Forecast'] = forecast

# 5. 计算预测的均方误差 (MSE)
mse = mean_squared_error(test_data['Sales'], test_data['Forecast'])
print(f'Mean Squared Error: {mse:.2f}')

# 6. 结果可视化
plt.figure(figsize=(10, 5))
plt.plot(train_data, label='Training Data')
plt.plot(test_data['Sales'], label='Actual Sales', marker='o')
plt.plot(test_data['Forecast'], label='Forecasted Sales', linestyle='--', marker='x')
plt.title('ARIMA Sales Forecast (2023)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
