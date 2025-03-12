#em2 app 3-9-25
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import pytz

# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
HOME_RADIUS = 10
MOON_RADIUS = 50
MOON_POSITION = (100, 100)  # Upper left corner
HOME_POSITION = (IMAGE_WIDTH - 150, IMAGE_HEIGHT - 100)  # Lower right corner
HORIZON_Y = HOME_POSITION[1]  # Align horizon with Home
EARTH_MOON_DISTANCE_KM = 384400  # Actual distance in kilometers
DAILY_DISTANCE_INCREMENT = 5  # km per day
START_DATE = datetime(2025, 1, 29)
START_DAYS = 137
TIMEZONE = "America/Chicago"  # St. Louis timezone

font_path = "Arial.ttf"
bgColor = "lightblue"

set_message= "看起来，跑得越慢心率越高？或者你跑得慢，是因为心率比较高？有趣，跟想象的正好相反。而且跑得越慢，心率越高，睡眠，脾胃都比较差。3月11日。"
# Calculate updated Days and Distance based on the current date
current_time = datetime.now(pytz.timezone(TIMEZONE))
days_elapsed = (current_time.date() - START_DATE.date()).days
if current_time.hour >= 6:  # Update only after 6:00 AM St. Louis time
    days = START_DAYS + days_elapsed
else:
    days = START_DAYS + days_elapsed - 1

distance = days * DAILY_DISTANCE_INCREMENT  # Distance in km

# Total days needed to reach the Moon (assuming 5 km/day)
max_days = EARTH_MOON_DISTANCE_KM / DAILY_DISTANCE_INCREMENT
print(max_days)

#days = 18250 #50 years need to multiply by 4.22 to make it to 76880 max_days
# Calculate runner position with polynomial scaling
t = (days * 4.22 / max_days)  # 50 years will be 91,250 km, have to make it to the moon == 384,400
runner_x = int((1 - t) * HOME_POSITION[0] + t * MOON_POSITION[0])
runner_y = int((1 - t) * HOME_POSITION[1] + t * MOON_POSITION[1])

print(days, 1- t)

# Create an image
image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "lightblue")
draw = ImageDraw.Draw(image)

# Draw the horizon at Home's level
draw.line([(0, HORIZON_Y), (IMAGE_WIDTH, HORIZON_Y)], fill="blue", width=2)

# Draw the Moon
draw.ellipse(
    [
        (MOON_POSITION[0] - MOON_RADIUS, MOON_POSITION[1] - MOON_RADIUS),
        (MOON_POSITION[0] + MOON_RADIUS, MOON_POSITION[1] + MOON_RADIUS),
    ],
    #fill="#ADD8E6",
    fill="#4444FF",
    outline="yellow",
)

# Draw Home
draw.ellipse(
    [
        (HOME_POSITION[0] - HOME_RADIUS, HOME_POSITION[1] - HOME_RADIUS),
        (HOME_POSITION[0] + HOME_RADIUS, HOME_POSITION[1] + HOME_RADIUS),
    ],
    fill="red",
    outline="black",
)

# Draw a line connecting Home and the Moon
draw.line([MOON_POSITION, HOME_POSITION], fill="orange", width=2)

# Draw the runner (small circle)
RUNNER_RADIUS = 8
draw.ellipse(
    [
        (runner_x - RUNNER_RADIUS, runner_y - RUNNER_RADIUS),
        (runner_x + RUNNER_RADIUS, runner_y + RUNNER_RADIUS),
    ],
    fill="green",
    outline="black",
)

#################
# Add label above the runner
#label_text = f"Date: {current_time.strftime('%Y-%m-%d')}\nDays: {days}\nDistance: {distance} km"
#draw.text((runner_x - 30, runner_y - 60), label_text, fill="navy")  # Centered above the runner

# Define the font and size
#font = ImageFont.truetype("arial.ttf", 18)  # Adjust size as needed
label_text = f"Date: {current_time.strftime('%Y-%m-%d')}\nDays: {days}\nDistance: {distance} km"
#draw.text((runner_x - 50, runner_y - 100), label_text, fill="navy", font=font)

moon_text = ("We choose to go to the Moon \n"
             "in this lifetime (and beyond?), \n"
             "not because it is easy,  \n"
             "but because it is hard :)")

#font_path = "Arial.ttf"

font_size = 24
font = ImageFont.truetype(font_path, font_size)
font2 = ImageFont.truetype(font_path, 20)

#draw.text((runner_x - 50, runner_y - 100), label_text, fill="navy", font=font)
#draw.text((runner_x - 300, runner_y - 200), moon_text, fill="navy",font= font)  # Cent
draw.text((runner_x - 25, runner_y - 100), label_text, fill="navy",font = font2)  # Centered above the runner

runa, runb = (runner_x - 300), ( runner_y - 200) # Example position

# Simulate bold by drawing multiple times with slight offsets
for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:  
    draw.text(( runa- 20 + offset[0], runb - 20 + offset[1]), 
             moon_text, fill="navy", font=font)

###more bold with the below
#for offset in [(0, 0), (1.5, 0), (0, 1.5), (1.5, 1.5)]:  
    #draw.text(( runa- 50 + offset[0], runb - 50 + offset[1]), 
              #moon_text, fill="navy", font=font)
# Add a small arrow pointing down from the label
arrow_start = (runner_x, runner_y - 20)  # Start of the arrow (just below the label)
arrow_end = (runner_x, runner_y-10)        # End of the arrow (pointing to the runner)
draw.line([arrow_start, arrow_end], fill="navy", width=2)  # Draw the arrow line

# Optional: Add arrowhead (a small triangle)
arrowhead_size = 5
arrowhead = [
    (runner_x - arrowhead_size, runner_y - 2* arrowhead_size),  # Left point
    (runner_x + arrowhead_size, runner_y - 2* arrowhead_size),  # Right point
    (runner_x, runner_y-arrowhead_size)                                     # Tip of the arrow
]
draw.polygon(arrowhead, fill="navy")  # Draw the arrowhead
############

# Add labels directly above Home and Moon
draw.text((HOME_POSITION[0] - 10, HOME_POSITION[1] +10), "Home", fill="black")  # Above Home
draw.text((MOON_POSITION[0] - 30, MOON_POSITION[1] - 80), "Moon\n384,400km", fill="black")  # Above Moon

# Display the image in Streamlit
st.title("Earth to Moon Running Visualization")
st.image(image, caption="A young man running from Home to the Moon 2075(2025) will be 91,250 km", use_container_width=True)
#sssss##############

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import streamlit as st
import pandas as pd
import matplotlib.font_manager as fm
import pytz  # For timezone handling
import textwrap

# Define your local timezone
LOCAL_TIMEZONE = pytz.timezone('America/Chicago')  # Replace with your local timezone

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
def save_data(data, data1, data2, data3, data4, filename="data.csv"):
    df = pd.DataFrame({
        "耳鸣级数": [",".join(map(str, data))],
        "脾胃": [",".join(map(str, data1))],
        "睡眠质量": [",".join(map(str, data2))],
        "心率": [",".join(map(str, data3))],
        "五千米时间": [",".join(map(str, data4))]
    })
    df.to_csv(filename, index=False)

# Function to load data from a CSV file
def load_data(filename="data.csv"):
    try:
        df = pd.read_csv(filename)
        data = [float(x) for x in df["耳鸣级数"].iloc[0].split(",")]
        data1 = [float(x) for x in df["脾胃"].iloc[0].split(",")]
        data2 = [float(x) for x in df["睡眠质量"].iloc[0].split(",")]
        data3 = [float(x) for x in df["心率"].iloc[0].split(",")]
        data4 = [float(x) for x in df["五千米时间"].iloc[0].split(",")]
        return data, data1, data2,data3,data4
    except FileNotFoundError:
        return None, None, None,None,None

# Streamlit App
st.title("健康数据分析")

# Load saved data (if it exists)
data, data1, data2, data3, data4 = load_data()

# Data Entry Page
st.sidebar.header("数据输入")
st.sidebar.write("请在下方输入或上传数据集。")

# Add a date input widget in the sidebar for selecting the start date
start_date = datetime(2024, 10, 22)
#start_date = st.sidebar.date_input("选择开始日期", datetime(2024, 10, 22))

# Input fields for datasets
st.sidebar.subheader("耳鸣级数")
er_ming_input = st.sidebar.text_area("输入耳鸣级数数据（逗号分隔）：", value=",".join(map(str, data)) if data else "2.0, 3.0, 2.5, 2.5, 2.0, 1.5, 2.5, 1.5, 2.0, 1.5, 1.5, 3.0, 1.5, 2.5, 1.5, 1.5, 1.5, 2.5, 1.5, 2.0, 2.0, 1.5, 2.5, 1.5, 1.5, 2.5, 1.5, 2.0, 1.5, 2.0, 1.5, 1.5, 2.0, 1.5, 2.5, 1.5, 2.5, 1.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.25, 1.25, 1.5, 1.5, 2.5, 1.5, 2.0, 1.5, 1.25, 1.5, 1.5, 2.0, 1.25, 1.5, 1.25, 1.25, 1.5, 1.25, 2.0, 1.25, 2.0, 1.25, 1.5, 1.25, 1.25, 1.5, 1.25, 2.0, 1.25")

st.sidebar.subheader("脾胃")
pi_wei_input = st.sidebar.text_area("输入脾胃数据（逗号分隔）：", value=",".join(map(str, data1)) if data1 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0, 0.5, 0.5, 0.5, 0.50, -0.5, 0.5, 0, 0, 0.5, 0.5, -0.5, 0.5, 0.0, -0.5, 0.5, 0.5, 0.0, 0, 0.5, 0.5, 0.25, 0.25, 0.5, 1.0, 0.25, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5")

st.sidebar.subheader("睡眠质量")
sleep_input = st.sidebar.text_area("输入睡眠质量数据（逗号分隔）：", value=",".join(map(str, data2)) if data2 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.2, 0.2, -0.2, 0, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0, 0.25")

st.sidebar.subheader("心率")
xinlv_input = st.sidebar.text_area("输入心率数据（逗号分隔）：", value=",".join(map(str, data3)) if data3 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.2, 0.2, -0.2, 0, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0, 0.25")

st.sidebar.subheader("5K时间")
wqshijian_input = st.sidebar.text_area("输入5K时间数据（逗号分隔）：", value=",".join(map(str, data4)) if data4 else "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.2, 0.2, -0.2, 0, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.5, 0.0, 0.25")


# Parse input data
data = parse_input(er_ming_input)
data1 = parse_input(pi_wei_input)
data2 = parse_input(sleep_input)
data3 = parse_input(xinlv_input)
data3= [x /16 for x in data3] 
data4 = parse_input(wqshijian_input)
data4 = [x /10 for x in data4] 

# Add a "key" input box for automatic saving
st.sidebar.subheader("自动保存设置")
key_input = st.sidebar.text_input("输入密钥以自动保存数据:", type="password")  # Hide the entered key

# Automatically save data if the key is "zzzzzzzzz"
if key_input.strip() == "zzzzzzzzz":
    if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
        save_data(data, data1, data2, data3, data4)
        st.sidebar.success("数据已自动保存！")
    else:
        st.sidebar.error("无法保存数据，请检查输入格式。")

# Add a button to download the saved data as a CSV file
if st.sidebar.button("下载数据为CSV文件"):
    if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
        save_data(data, data1, data2, data3, data4, "health_data.csv")
        with open("health_data.csv", "rb") as file:
            st.sidebar.download_button(
                label="点击下载CSV文件",
                data=file,
                file_name="health_data.csv",
                mime="text/csv"
            )
    else:
        st.sidebar.error("没有可下载的数据，请先输入数据并保存。")

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
    ma_data5 = moving_average(data3)##
    ma_data6 = moving_average(data4)###

    # 耳鸣级数动态均值
    data0 = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] + data
    new_data = data[6:]
    data10= moving_average(data0)
    new_data4 = data10[6:]
    ma_data4 = moving_average(data10)

    # 脾胃动态均值
    new_data1 = data1[6:]

    # 睡眠质量动态均值
    new_data2 = data2[6:]

    new_data5 = data3[6:]##

    new_data6 = data4[6:]###

    # Define the last date and calculate the start date
    # Use the selected start_date from the sidebar
    num_points = len(ma_data)  # Assume all datasets have the same number of points in the moving average
    new_date = start_date + timedelta(days=num_points - 1)
    last_date = new_date.strftime("%b %d, %Y")

    # Plotting
    fm.fontManager.addfont('SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or Arial Unicode MS
    plt.rcParams['axes.unicode_minus'] = False

    datasets = [new_data5,new_data6, new_data, new_data1, new_data2 ]
    ma_datasets = [ma_data5,  ma_data6, ma_data, ma_data1, ma_data2 ]
    titles = ["心率动态均值（最高值百分比/10）", "5K慢跑时间动态均值（/10）" , "耳鸣级数动态均值（最高： 6）", "脾胃动态均值（最高：1）", "睡眠质量动态均值（最高：1）" ]
    colors = ['blue', 'green', 'red', 'blue','green']

# Assuming bgColor, start_date, and other variables are already defined
    labels = ["心率(最高值百分比/10）", "5K慢跑分钟/10", "耳鸣级数", "脾胃", "睡眠质量"]

    fig, ax = plt.subplots(figsize=(12, 15))  # Single figure for combined plot

    for i in range(5):  # Loop through the first 5 datasets
        trimmed_data = datasets[i][:len(ma_datasets[i])]  # Trim original data to match moving average length
    
    # Plot original data
        #ax.plot(trimmed_data, label=f"{labels[i]} 原始数据", color=colors[i], linestyle='--', alpha=0.7)
        ax.plot(trimmed_data, color=colors[i], linestyle='--', alpha=0.7)
    
    # Plot moving average
        #ax.plot(ma_datasets[i], label=f"{labels[i]} 7天动态均值", color=colors[i])
        ax.plot(ma_datasets[i], color=colors[i])
    # Highlight last point of moving average
        ax.scatter(len(ma_datasets[i]) - 1, ma_datasets[i][-1], color=colors[i])
    
    # Add text annotation for the last point
        last_date = start_date + timedelta(days=len(ma_datasets[i]) - 1)
        ax.text(len(ma_datasets[i]) +15, ma_datasets[i][-1], 
            f'{last_date.strftime("%m-%d")} ({ma_datasets[i][-1]:.2f})', 
            color='black', fontsize=10, ha='right')
        
        # Set Y-axis ticks to step 1
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

 ##########. add labels separately
    ax.text(10, 7.8,
        "心率（最高值百分比/10）（小米）",
        color='blue', fontsize=15, ha='left')   
    ax.text(10, 6.2,
        "5K慢跑分钟/10（小米）",
        color='green', fontsize=15, ha='left')   
    ax.text(145, 2.0,
        "耳鸣级数",
        color='red', fontsize=15, ha='right')   
    ax.text(145, 0.5,
        "脾胃",
        color='blue', fontsize=15, ha='right')   
    ax.text(145, -0.3,
        "睡眠",
        color='green', fontsize=15, ha='right')   

    # Use Streamlit sliders to adjust line positions
    h_line_pos = st.sidebar.slider('Horizontal Line Position', - 1, 9, 1)
    v_line_pos = st.sidebar.slider('Vertical Line Position', 5 , 140, 140)
    
    #st.sidebar.subheader("marker_message")
    marker_message_input = st.sidebar.text_area("输入分析信息：", value=set_message)
    # Function to wrap text into lines of a specified width
    def wrap_text(text, width):
        return "\n".join(textwrap.wrap(text, width=width))

# Position and content for the text annotation
    wrapped_text = wrap_text(marker_message_input, 12)  # Wrap text to 25 characters per line

    ax.text(v_line_pos, 3.5,
        wrapped_text,
        color='navy', fontsize=16, ha='right') 

    # Add lines based on slider values
    ax.axhline(y=h_line_pos, color='orange', lw=2, linestyle='--')
    ax.axvline(x=v_line_pos, color='orange', lw=2, linestyle='--')

# #### Set background color
    ax.set_facecolor(bgColor)

# Add title, labels, legend, and grid
    ax.set_title("综合动态均值分析(虚线原始数据）")
    ax.set_xlabel("天数")
    ax.set_ylabel("动态均值（7天均值）")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Place legend outside the plot
    ax.grid()

# Display the combined plot
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
    current_date = datetime.now(LOCAL_TIMEZONE)
    future_date = current_date + timedelta(days=prediction_days)
    future_date_str = future_date.strftime("%b %d, %Y")
    last_linear_prediction = linear_future_predictions[-1]
    last_poly_prediction = poly_future_predictions[-1]
    
    # Plot the trend analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    
########
    LOCAL_TIMEZONE = pytz.timezone('America/Chicago') 
    today = datetime.now(LOCAL_TIMEZONE)
    print(today)

    # Define the past date and make it timezone-aware
    past_date = LOCAL_TIMEZONE.localize(datetime(2024, 9, 14))
    print("Past Date:", past_date)

    # Calculate the difference in days
    delta = today - past_date
    days_difference = delta.days
    print("Days Difference:", days_difference)
    
    # Add the current date
    current_tinnitus_level = data[-1]  # Latest value in the tinnitus data
    current_double_ma_tinnitus_level = double_ma_data[-1]  # Latest value in the double moving average data
    
    current_date = datetime.now(LOCAL_TIMEZONE).strftime("%Y年%m月%d日")
    
    # Display the current date and values in the plot
    
    ax.text(0.3, 0.85, f"慢跑第{days_difference}天\n\n{current_date} \n耳鸣级数：{current_tinnitus_level:.2f}\n双动态均值：{current_double_ma_tinnitus_level:.2f}", 
        horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=12, color="blue")
  
    ax.scatter(time_steps, double_ma_data, color="blue", label="双动态均值（7日）")
    ax.plot(time_steps, linear_model.predict(time_steps), color="red", label="线性回归")
    ax.plot(time_steps, poly_model.predict(time_steps), color="green", label="多项式回归 (2次)")
    ax.scatter(future_time_steps, linear_future_predictions, color="orange", label=f"线性预测 ({prediction_days} 天)")
    ax.scatter(future_time_steps, poly_future_predictions, color="purple", label=f"多项式预测 ({prediction_days} 天)")

    # Add arrow pointing to the last predicted point
    
    #add arrow to last point of double_ma_data
    ax.annotate(
        f"(双动态均值）",
        xy=(time_steps[-1], current_double_ma_tinnitus_level),
        xytext=(time_steps[-1], current_double_ma_tinnitus_level + 0.2),
        arrowprops=dict(facecolor='blue', shrink=0.05, headwidth=10, width=3),
        fontsize=10,
        color="red",
        ha="center"
    )
    
    ax.annotate(
        f"{future_date_str}: {last_linear_prediction:.2f} (线性)",
        xy=(future_time_steps[-1], last_linear_prediction),
        xytext=(future_time_steps[-1], last_linear_prediction + 0.1),
        arrowprops=dict(facecolor='orange', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="red",
        ha="center"
    )

    ax.annotate(
        f"{future_date_str}: {last_poly_prediction:.2f} (多项式)",
        xy=(future_time_steps[-1], last_poly_prediction),
        xytext=(future_time_steps[-1], last_poly_prediction + 0.1),
        arrowprops=dict(facecolor='purple', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="green",
        ha="center"
    )

    # Add the last 28 points of data to the plot no no no
    last_21_days_data = data[-60:]
    last_21_days_time_steps = np.arange(len(double_ma_data) -60, len(double_ma_data)).reshape(-1, 1)

    ax.scatter(last_21_days_time_steps, last_21_days_data, color="gray", label="最近8周耳鸣级数", marker=".", s=100)
    ax.set_facecolor(bgColor)
    # Add labels and legend
    ax.set_xlabel("天数")
    ax.set_ylabel("耳鸣级数")
    ax.set_title(f"耳鸣级数双动态均值 线性回归与多项式回归分析 + {prediction_days}天预测")
    ax.legend()
    ax.grid()

    ######
    # Calculate the R² values
    r2_linear = linear_model.score(time_steps, double_ma_data)
    r2_poly = poly_model.score(time_steps, double_ma_data)

    # Add the linear model R² label in blue
    ax.text(0.02, 0.10, f"- $R^2$ (线性回归): {r2_linear:.3f}", transform=ax.transAxes, fontsize=10,
        color='red', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # Add the polynomial model R² label in green
    ax.text(0.02, 0.02, f"- $R^2$ (多项式回归): {r2_poly:.3f}", transform=ax.transAxes, fontsize=10,
        color='green', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

# Show the plot in Streamlit

    st.pyplot(fig)


st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("=============================================")




