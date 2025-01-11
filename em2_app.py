import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import streamlit as st
import pandas as pd
import matplotlib.font_manager as fm

# Set matplotlib font to support Chinese characters
fm.fontManager.addfont('SimHei.ttf') 
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False

# Function to calculate 7-point moving average
def moving_average(data, window_size=7):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to parse input data
def parse_input(input_data):
    try:
        # Convert input string to a list of floats
        data = [float(x.strip()) for x in input_data.split(",")]
        return data
    except ValueError:
        st.error("Invalid input! Please enter comma-separated numbers.")
        return None

# Function to save data to a CSV file
def save_data(data, data1, data2, filename="data.csv"):
    df = pd.DataFrame({
        "耳鸣级数": [",".join(map(str, data))],
        "脾胃": [",".join(map(str, data1))],
        "睡眠质量": [",".join(map(str, data2))]
    })
    df.to_csv(filename, index=False)

# Function to load data from a CSV file
def load_data(filename="data.csv"):
    try:
        df = pd.read_csv(filename)
        data = [float(x) for x in df["耳鸣级数"].iloc[0].split(",")]
        data1 = [float(x) for x in df["脾胃"].iloc[0].split(",")]
        data2 = [float(x) for x in df["睡眠质量"].iloc[0].split(",")]
        return data, data1, data2
    except FileNotFoundError:
        return None, None, None

# Streamlit App
st.title("健康数据分析")

# Load saved data (if it exists)
data, data1, data2 = load_data()

# Data Entry Page
st.sidebar.header("数据输入")
st.sidebar.write("请在下方输入或上传数据集。")

# Add a date input widget in the sidebar for selecting the start date
start_date = st.sidebar.date_input("选择开始日期", datetime(2024, 10, 22))

# Input fields for datasets
st.sidebar.subheader("耳鸣级数")
er_ming_input = st.sidebar.text_area("输入耳鸣级数数据（逗号分隔）：", value=",".join(map(str, data)) if data else "2.0, 3.0, 2.5, 2.5, 2.0, 1.5, 2.5, 1.5, 2.0, 1.5, 1.5, 3.0, 1.5, 2.5, 1.5, 1.5, 1.5, 2.5, 1.5, 2.0, 2.0, 1.5, 2.5, 1.5, 1.5, 2.5, 1.5, 2.0, 1.5, 2.0, 1.5, 1.5, 2.0, 1.5, 2.5, 1.5, 2.5, 1.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.25, 1.25, 1.5, 1.5, 2.5, 1.5, 2.0, 1.5, 1.25, 1.5, 1.5, 2.0, 1.25, 1.5, 1.25, 1.25, 1.5, 1.25, 2.0, 1.25, 2.0, 1.25, 1.5, 1.25, 1.25, 1.5, 1.25, 2.0, 1.25")

st.sidebar.subheader("脾胃")
pi_wei_input = st.sidebar.text_area("输入脾胃数据（逗号分隔）：", value=",".join(map(str, data1)) if data1 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0, 0.5, 0.5, 0.5, 0.50, -0.5, 0.5, 0, 0, 0.5, 0.5, -0.5, 0.5, 0.0, -0.5, 0.5, 0.5, 0.0, 0, 0.5, 0.5, 0.25, 0.25, 0.5, 1.0, 0.25, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5")

st.sidebar.subheader("睡眠质量")
sleep_input = st.sidebar.text_area("输入睡眠质量数据（逗号分隔）：", value=",".join(map(str, data2)) if data2 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.2, 0.2, -0.2, 0, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0, 0.25")

# Parse input data
data = parse_input(er_ming_input)
data1 = parse_input(pi_wei_input)
data2 = parse_input(sleep_input)

# Save data button
if st.sidebar.button("保存数据"):
    if data is not None and data1 is not None and data2 is not None:
        save_data(data, data1, data2)
        st.sidebar.success("数据已保存！")
    else:
        st.sidebar.error("无法保存数据，请检查输入格式。")

# Add a divider before the new section
st.sidebar.markdown("---")

# Add the title for the new section
st.sidebar.header("耳鸣级数参考方法")

# Add the image (replace 'path_to_your_image.png' with the actual path to your image)
st.sidebar.image("erming_jishu.png", caption="耳鸣级数参考图", use_container_width=True)

# Check if data is valid
if data is not None and data1 is not None and data2 is not None:
    # Calculate moving averages
    ma_data = moving_average(data)
    ma_data1 = moving_average(data1)
    ma_data2 = moving_average(data2)

    # 耳鸣级数动态均值
    data0 = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] + data
    new_data = data[6:]
    data4 = moving_average(data0)
    new_data4 = data4[6:]
    ma_data4 = moving_average(data4)

    # 脾胃动态均值
    new_data1 = data1[6:]

    # 睡眠质量动态均值
    new_data2 = data2[6:]

    # Define the last date and calculate the start date
    # Use the selected start_date from the sidebar
    num_points = len(ma_data)  # Assume all datasets have the same number of points in the moving average
    new_date = start_date + timedelta(days=num_points - 1)
    last_date = new_date.strftime("%b %d, %Y")

    # Plotting
    fm.fontManager.addfont('SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or Arial Unicode MS
    plt.rcParams['axes.unicode_minus'] = False

    datasets = [new_data, new_data1, new_data2, new_data4]
    ma_datasets = [ma_data, ma_data1, ma_data2, ma_data4]
    titles = ["耳鸣级数动态均值（最高： 6）", "脾胃动态均值（最高：1）", "睡眠质量动态均值（最高：1）", "耳鸣级数双动态均值（最高： 6）"]
    colors = ['blue', 'green', 'red', 'blue']

    for i in range(4):
        fig, ax = plt.subplots(figsize=(10, 4))
        trimmed_data = datasets[i][:len(ma_datasets[i])]  # Trim original data to match moving average length
        ax.plot(trimmed_data, label="原始数据", color='orange', linestyle='--', alpha=0.7)
        ax.plot(ma_datasets[i], label="7天动态均值", color=colors[i])
        ax.scatter(len(ma_datasets[i]) - 1, ma_datasets[i][-1], color=colors[i])  # Highlight last point
        
        # Calculate date for the last data point
        last_date = start_date + timedelta(days=len(ma_datasets[i]) - 1)
        ax.text(len(ma_datasets[i]) - 1, ma_datasets[i][-1], 
                f'{last_date.strftime("%m-%d")} ({ma_datasets[i][-1]:.2f})', 
                color='red', fontsize=10)
        
        ax.set_title(titles[i])
        ax.set_xlabel("天数")
        ax.set_ylabel(titles[i])
        ax.legend()
        ax.grid()
        
        st.pyplot(fig)

    # Reformat last_date string    
    last_date = new_date.strftime("%b %d, %Y")
    
    # Display last 30-day average
    dataave30 = np.mean(data[-30:])
    st.write(f"耳鸣级数最近30天平均值: {dataave30:.2f}")

    ###################### Trend analysis #############

    # Page break
    st.markdown("---")
    
    # Trend Analysis: Double Moving Averages and Linear Regression
    st.header("耳鸣级数双动态均值趋势分析")

    # Add a sidebar widget to select prediction days
    st.sidebar.subheader("选择预测天数")
    prediction_option = st.sidebar.radio("选择预测选项:", ["30天", "60天", "90天", "自定义天数"])
    
    if prediction_option == "自定义天数":
        prediction_days = st.sidebar.number_input("输入预测天数:", min_value=1, value=30)
    else:
        prediction_days = int(prediction_option.replace("天", ""))

    # Double moving averages
    double_ma_data = np.array(ma_data4)

    # Create time steps (independent variable)
    time_steps = np.arange(len(double_ma_data)).reshape(-1, 1)

    # Fit the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(time_steps, double_ma_data)

    # Fit the polynomial regression model (degree=2 for quadratic)
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(time_steps, double_ma_data)

    # Create time steps for the future based on user selection
    future_time_steps = np.arange(len(double_ma_data), len(double_ma_data) + prediction_days).reshape(-1, 1)

    # Predict future values using both models
    linear_future_predictions = linear_model.predict(future_time_steps)
    poly_future_predictions = poly_model.predict(future_time_steps)

    # Get the date for the future prediction and last predicted value
    current_date = datetime.now()
    future_date = current_date + timedelta(days=prediction_days)
    future_date_str = future_date.strftime("%b %d, %Y")
    last_linear_prediction = linear_future_predictions[-1]
    last_poly_prediction = poly_future_predictions[-1]
    
    # Plot the trend analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_steps, double_ma_data, color="blue", label="原始数据")
    ax.plot(time_steps, linear_model.predict(time_steps), color="red", label="线性回归")
    ax.plot(time_steps, poly_model.predict(time_steps), color="green", label="多项式回归 (2次)")
    ax.scatter(future_time_steps, linear_future_predictions, color="orange", label=f"线性预测 ({prediction_days} 天)")
    ax.scatter(future_time_steps, poly_future_predictions, color="purple", label=f"多项式预测 ({prediction_days} 天)")

    # Add arrow pointing to the last predicted point
    ax.annotate(
        f"{future_date_str}: {last_linear_prediction:.2f} (线性)",
        xy=(future_time_steps[-1], last_linear_prediction),
        xytext=(future_time_steps[-1], last_linear_prediction + 0.1),
        arrowprops=dict(facecolor='red', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="red",
        ha="center"
    )

    ax.annotate(
        f"{future_date_str}: {last_poly_prediction:.2f} (多项式)",
        xy=(future_time_steps[-1], last_poly_prediction),
        xytext=(future_time_steps[-1], last_poly_prediction + 0.1),
        arrowprops=dict(facecolor='green', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="green",
        ha="center"
    )

    # Add labels and legend
    ax.set_xlabel("天数")
    ax.set_ylabel("耳鸣级数")
    ax.set_title(f"耳鸣级数双动态均值 线性回归与多项式回归分析 + {prediction_days}天预测")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    # Display regression results
    st.write(f"线性回归分析结果:")
    st.write(f"- 斜率 (m): {linear_model.coef_[0]:.4f}")
    st.write(f"- 截距 (b): {linear_model.intercept_:.4f}")
    st.write(f"- R²: {linear_model.score(time_steps, double_ma_data):.3f}")
    st.write(f"- {prediction_days}天预测值 ({prediction_days}天后， {future_date_str}): {last_linear_prediction:.2f}")

    st.write(f"多项式回归分析结果:")
    st.write(f"- R²: {poly_model.score(time_steps, double_ma_data):.3f}")
    st.write(f"- {prediction_days}天预测值 ({prediction_days}天后， {future_date_str}): {last_poly_prediction:.2f}")       
