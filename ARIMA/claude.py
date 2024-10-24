import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# 生成真实的销售数据
def generate_sales_data():
    # 设定随机种子以确保可重复性
    np.random.seed(42)

    # 创建日期范围: 2022-01-01 到 2023-12-31
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

    # 基础销量
    base_sales = 1000

    # 生成销量数据
    sales = []
    for i in range(len(dates)):
        # 添加季节性波动 (使用正弦函数)
        seasonal = 200 * np.sin(2 * np.pi * i / 365)

        # 添加周末效应
        weekend_effect = 300 if dates[i].weekday() >= 5 else 0

        # 添加假日效应 (元旦、春节、五一、国庆)
        holiday_effect = 0
        if dates[i].strftime('%m-%d') in ['01-01', '02-01', '05-01', '10-01']:
            holiday_effect = 500

        # 添加趋势增长
        trend = i * 0.5

        # 添加随机波动
        noise = np.random.normal(0, 50)

        # 合并所有效应
        daily_sales = base_sales + seasonal + weekend_effect + holiday_effect + trend + noise

        # 确保销量为正数
        daily_sales = max(0, daily_sales)
        sales.append(daily_sales)

    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })

    return df


# 训练ARIMA模型并进行预测
def train_and_forecast(data, forecast_steps=30):
    # 拟合ARIMA模型
    model = ARIMA(data['sales'], order=(5, 1, 2))
    results = model.fit()

    # 进行预测
    forecast = results.forecast(steps=forecast_steps)

    # 创建预测日期
    last_date = data['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                   periods=forecast_steps,
                                   freq='D')

    # 创建预测DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast
    })

    return results, forecast_df


# 可视化结果
def plot_results(data, forecast_df):
    plt.figure(figsize=(15, 7))

    # 绘制历史数据
    plt.plot(data['date'], data['sales'], label='Historical Sales', color='blue')

    # 绘制预测数据
    plt.plot(forecast_df['date'], forecast_df['forecast'],
             label='Forecasted Sales', color='red', linestyle='--')

    plt.title('Sales Forecast using ARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)

    # 旋转x轴日期标签以防重叠
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    plt.savefig('sales_forecast.png')
    plt.close()


# 主函数
def main():
    # 生成数据
    print("生成销售数据...")
    sales_data = generate_sales_data()

    # 训练模型和预测
    print("训练ARIMA模型并进行预测...")
    model_results, forecast_data = train_and_forecast(sales_data)

    # 打印模型摘要
    print("\nARIMA模型摘要:")
    print(model_results.summary())

    # 绘制结果
    print("\n绘制预测结果...")
    plot_results(sales_data, forecast_data)

    # 打印预测结果
    print("\n未来30天的销量预测:")
    print(forecast_data.to_string())

    # 保存数据到CSV文件
    sales_data.to_csv('historical_sales.csv', index=False)
    forecast_data.to_csv('sales_forecast.csv', index=False)

    print("\n数据已保存到 'historical_sales.csv' 和 'sales_forecast.csv'")
    print("预测图表已保存为 'sales_forecast.png'")


if __name__ == "__main__":
    main()