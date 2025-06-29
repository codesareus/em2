#em2 app 3-19-25
import streamlit as st
from gtts import gTTS
from datetime import datetime, timedelta
import pytz
import re  # Import regex
import os
import csv

import matplotlib.pyplot as plt
        
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import pytz
import numpy as np
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.font_manager as fm
import textwrap
from sklearn.metrics import r2_score

##################################

#divider image
#imageName = "cherry.jpeg"
#st.image("cherry.jpeg",width=705)

#st.image("cherry.jpeg",width=710)


##############。 耳鸣分析
# Set matplotlib font to support Chinese characters
#fm.fontManager.addfont('SimHei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False

# Function to calculate 7-point moving average#########
#def moving_average(data, window_size=4):
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

st.sidebar.subheader("每日观察")
thought_input = st.sidebar.text_area("", value="")
 #st.sidebar.subheader("marker_message")
marker_message_input = st.sidebar.text_area("输入分析信息：", value="今天耳鸣特别好，做了啥？ @######\n" )
    
# Data Entry Page
st.sidebar.header("数据输入")
st.sidebar.write("请在下方输入或上传数据集。")

# Load saved data (if it exists)
data, data1, data2, data3, data4 = load_data()

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
xinlv_input = st.sidebar.text_area("输入心率数据（逗号分隔）：", value=",".join(map(str, data3)) if data3 else "")

st.sidebar.subheader("5K时间")
wqshijian_input = st.sidebar.text_area("输入5K时间数据（逗号分隔）：", value=",".join(map(str, data4)) if data4 else "")

##############
from datetime import datetime, timedelta
import pytz  # Needed for timezone handling
import matplotlib.font_manager as fm

# Add the SimHei font to Matplotlib's font manager
fm.fontManager.addfont('SimHei.ttf')

st.title("Plot the Last 21 Daily Data Points with Moving Average (Central Time)")

# Configure Central Time zone
cst = pytz.timezone('America/Chicago')

def get_cst_date():
    """Get current date in US Central Time"""
    now_utc = datetime.now(pytz.utc)
    return now_utc.astimezone(cst).date()


# Parse input data
data = parse_input(er_ming_input)
data1 = parse_input(pi_wei_input)
data2 = parse_input(sleep_input)
dataa = parse_input(xinlv_input)
data3= [round(x /1.6, 1) for x in dataa] 
datab = parse_input(wqshijian_input)
data4 = [round(x /1.0, 1) for x in datab] 

dataSets =[data, data1, data2, data4, data3]

units =  ['级','（最高1.0)','（最高1.0)','分钟','(最高值百分比)']

LOCAL_TIMEZONE = pytz.timezone('America/Chicago') 
today = datetime.now(LOCAL_TIMEZONE)
    

    # Define the past date and make it timezone-aware
past_date = LOCAL_TIMEZONE.localize(datetime(2024, 9, 14))
    
# Calculate the difference in days
delta = today - past_date
days_difference = delta.days
    
current_date = datetime.now(LOCAL_TIMEZONE).strftime("%Y年%m月%d日")
message2 = f"慢跑第{days_difference}天, {current_date} >>>>> 耳鸣级数"

names = [message2,'脾胃','睡眠质量','5K时长(分钟)','5K心率均值(最高值百分比)']

##### plot 耳鸣 only

# Create a SINGLE plot figure
fig, ax = plt.subplots(figsize=(12.5, 5))  # Adjusted height for clarity
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Use only the FIRST dataset (index 0)
raw_data = dataSets[0]
title_name = names[0]
unit = units[0]

ax.set_facecolor('#e6f7ff')  # Light blue background

try:
    numbers = [float(x) for x in raw_data if x]
    if len(numbers) < 27:
        ax.text(0.5, 0.5, f"Not enough data ({len(numbers)} points)\nNeed at least 27", 
                ha='center', va='center', color='red')
        ax.set_title(title_name)
        ax.axis('on')
    else:
        # Generate dates
        today_cst = get_cst_date()
        dates = [today_cst - timedelta(days=i) for i in range(20, -1, -1)]  # Last 21 days
        last_27 = numbers[-27:]

        # Moving average
        moving_avg = [sum(last_27[i:i+7])/7 for i in range(21)]
        original_data = last_27[-21:]

        # Plotting
        ax.plot(dates, original_data, marker='o', label=title_name)
        ax.plot(dates, moving_avg, label='7-天动态均值', color='orange', linestyle='--', linewidth=2)

        ax.set_title(title_name)
        ax.set_xlabel(f"Date (Central Time)\n{today_cst.strftime('%Y-%m-%d')}({raw_data[-1]}{unit})")
        ax.set_ylabel(title_name)
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime("%m-%d\n%a") for d in dates], rotation=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')

except ValueError:
    ax.text(0.5, 0.5, "Invalid input: Only numeric values allowed", ha='center', va='center', color='red')
    ax.set_title(title_name)
    ax.axis('on')
except Exception as e:
    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    ax.set_title(title_name)
    ax.axis('on')

# Display the figure in Streamlit
st.pyplot(fig)


# Create a single figure with 4 subplots stacked vertically
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12.5, 2.5*5))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Increase vertical spacing between subplots to avoid overlap
plt.subplots_adjust(hspace=1)  # You can tweak this value as needed

for ax, raw_data, title_name, unit in zip(axes, dataSets, names, units):
    ax.set_facecolor('#e6f7ff')  # Light blue background
    try:
        numbers = [float(x) for x in raw_data if x]
        if len(numbers) < 27:
            ax.text(0.5, 0.5, f"Not enough data ({len(numbers)} points)\nNeed at least 27", 
                    ha='center', va='center', color='red')
            ax.set_title(title_name)
            ax.axis('on')
            continue

        # Generate dates
        today_cst = get_cst_date()
        dates = [today_cst - timedelta(days=i) for i in range(20, -1, -1)]  # Last 21 days
        last_27 = numbers[-27:]

        # Moving average
        moving_avg = [sum(last_27[i:i+7])/7 for i in range(21)]
        original_data = last_27[-21:]

        # Plotting
        ax.plot(dates, original_data, marker='o', label=title_name)
        ax.plot(dates, moving_avg, label='7-天动态均值', color='orange', linestyle='--', linewidth=2)

        ax.set_title(title_name)
        ax.set_xlabel(f"Date (Central Time)\n{today_cst.strftime('%Y-%m-%d')}({raw_data[-1]}{unit})")
        ax.set_ylabel(title_name)
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime("%m-%d\n%a") for d in dates], rotation=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')

    except ValueError:
        ax.text(0.5, 0.5, "Invalid input: Only numeric values allowed", ha='center', va='center', color='red')
        ax.set_title(title_name)
        ax.axis('on')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
        ax.set_title(title_name)
        ax.axis('on')

# Display the full figure in Streamlit
st.pyplot(fig)
st.write("==============")
st.write("==============")
###############
###############.    draw moon and runner
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


font_path = "Arial.ttf"
bgColor = "lightblue"

# 添加中文字体支持（示例使用微软雅黑，请确保系统有该字体）
# 也可以替换为其他中文字体路径如 "simhei.ttf"
chinese_font = ImageFont.truetype("SimHei.ttf", 12)  # 调整字体大小

fm.fontManager.addfont('SimHei.ttf')

set_message=""
# Calculate updated Days and Distance based on the current date
LOCAL_TIMEZONE = pytz.timezone('America/Chicago')  # Replace with your local timezone
current_time = datetime.now(LOCAL_TIMEZONE)

days_elapsed = (current_time.date() - START_DATE.date()).days
if current_time.hour >= 6:  # Update only after 6:00 AM St. Louis time
    days = START_DAYS + days_elapsed
else:
    days = START_DAYS + days_elapsed - 1

distance = days * DAILY_DISTANCE_INCREMENT+ 20# Distance in km

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

label_text = f"Date: {current_time.strftime('%Y-%m-%d')}\nDays: {days}\nDistance: {distance} km"
#draw.text((runner_x - 50, runner_y - 100), label_text, fill="navy", font=font)

moon_text = ("We choose to go to the Moon \n"
             "in this lifetime (and beyond?), \n"
             "not because it is easy,  \n"
             "but because it is hard :)")

em_text = "1 改掉不良生活习惯，重中之重，2 适当，足量的运动，让身体有个向健康发展的好基础，\n3 调整好自己的情绪，不嗔，4 多多照顾经络，按揉或者是艾灸，\n5 如果脾胃不好，应该检查一下，是否饮食上，即便多年没事的食物，有些可能的潜在问题。\n6.颈部运动，头/颈部经络，胆经，三焦经+肝经，肾经，脾经。__耳鸣提示"

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
# 绘制文本时指定字体
draw.text((MOON_POSITION[0] - 10, HOME_POSITION[1] +10), em_text, fill="black", font=chinese_font)

# Display the image in Streamlit
st.title("Earth to Moon Running Visualization")
st.image(image, caption="A young man running from Home to the Moon 2075(2025) will be 91,250 km", use_container_width=True)

###############

# Parse input data for original
data = parse_input(er_ming_input)
data1 = parse_input(pi_wei_input)
data2 = parse_input(sleep_input)
dataa = parse_input(xinlv_input)
data3= [round(x /16, 1) for x in dataa] 
datab = parse_input(wqshijian_input)
data4 = [round(x /10, 1) for x in datab] 

# Convert lists to comma-separated strings without brackets
# Convert lists to comma-separated strings without brackets
formatted_string = f'"{",".join(map(str, data))}","{",".join(map(str, data1))}","{",".join(map(str, data2))}","{",".join(map(str, dataa))}","{",".join(map(str, datab))}"'# Display the formatted string in a read-only text box
st.sidebar.subheader("所有输入数据")

## let user upload health data
# Let user upload health data through the sidebar
upload_content = "upload content here"
with st.sidebar:
    #st.header("Data Upload")
    # Add checkbox to toggle upload visibility
    show_upload = st.checkbox("Upload local health data?", value=False)
    
    if show_upload:
        uploaded_file = st.file_uploader("Choose txt file", type=["txt"])
        
        if uploaded_file is not None:
            upload_content = uploaded_file.read().decode('utf-8')
            # Split the content by commas to separate the series
            # Parse the content using csv.reader to handle quoted strings
            reader = csv.reader([upload_content], quotechar='"')
            parsed = next(reader)  # Extracts the two string parts

            # Convert each part into a list of floats
            result = []
            for part in parsed:
                numbers = [float(num.strip()) for num in part.split(',')]
                result.append(numbers)
            #save the uploaded data back to data.csv file
            save_data(result[0], result[1], result[2], result[3], result[4])
                
#st.sidebar.text_area("复制以下格式化字符串：", value=formatted_string, height=100, key="formatted_output")
with open('rebootdate.txt', "r") as file:
    reboot = file.read()

st.sidebar.code(formatted_string, language="plaintext")
st.sidebar.code(reboot,language="plaintext")
st.sidebar.code(upload_content)

# Add a "key" input box for automatic saving
st.sidebar.subheader("自动保存设置")
key_input = st.sidebar.text_input("输入密钥以自动保存数据:", type="default")  # Hide the entered key

# Automatically save data if the key is "zzzzzzzzz"
if key_input.strip() == "z":
    if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
        save_data(data, data1, data2, dataa, datab)
        st.sidebar.success("数据已自动保存！")
    else:
        st.sidebar.error("无法保存数据，请检查输入格式。")

# Add a download button for the formatted string
    st.sidebar.download_button(
        label="下载为文本文件",
        data=formatted_string,
        file_name="health_Data.txt",
        mime="text/plain"
    )

# Add a divider before the new section
st.sidebar.markdown("---") #day

# Add the title for the new section
st.sidebar.header("耳鸣级数参考方法")

# Add the image (replace 'path_to_your_image.png' with the actual path to your image)
st.sidebar.image("erming_jishu.png", caption="耳鸣级数参考图", use_container_width=True)


# Check if data is valid
if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
    # Calculate moving averages
    ma_data = moving_average(data)
    ma_data1 = moving_average(data1)
    ma_data2 = moving_average(data2)
    ma_data5 = moving_average(data3)##not conflicting with versions of data. original data3 and data4 OK, but deviatives should use different numbers
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

########## time matters
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
    
    # Use Streamlit sliders to adjust line positions
    h_line_pos = st.sidebar.slider('Horizontal Line Position', - 1, 9, 1)
    v_line_pos = st.sidebar.slider('Vertical Line Position', 5 , days_difference, days_difference -38)
    
   # Function to wrap text into lines of a specified width
   # def wrap_text(text, width):
       # return "\n".join(textwrap.wrap(text, width=width))
    

    def wrap_text(text):
    # Split text into sections using "@"
        sections = text.split("@")
    
    # Process each section:
        wrapped_lines = []
        for section in sections:
        # Strip whitespace and skip empty sections
            stripped = section.strip()
            if not stripped:
                continue
        
        # Wrap the section into 32-character lines
            wrapped = textwrap.wrap(stripped, width=38)
        
        # Add the wrapped lines to the result
            wrapped_lines.extend(wrapped)
    
    # Join all lines with single newlines
        return "\n".join(wrapped_lines)

# Position and content for the text annotation
    
    current_date = datetime.now(LOCAL_TIMEZONE).strftime("%Y年%m月%d日")
    message1 = f"慢跑第{days_difference}天, {current_date}: \n"
    wrapped_input = wrap_text(marker_message_input)  # Wrap text to 25 characters per line
    dayData = f"\n耳鸣：{data[-1]}, 脾胃：{data1[-1]}, 睡眠：{data2[-1]},慢跑心率（最高值百分比）：{data3[-1] *10}, 时长：{data4[-1]*10}分"
    
    ax.text(0.5, 2.5,
        message1 +set_message   + "\n"+ wrapped_input + "\n"+ dayData,
        color='navy', fontsize=16, ha='left', 
        bbox={
            'facecolor': '#FFCCCB',  # Light red (hex code for "light coral")
            'edgecolor': 'none',    # Remove border
            'boxstyle': 'round',     # Optional: rounded corners
            'pad': 0.5      }
            # Padding around text)
           )
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
    st.header("耳鸣级数双动态均值趋势分析")

    if "prdegree" not in st.session_state:
        st.session_state.prdegree = 3
    # Trend Analysis: Double Moving Averages and Linear Regression
    col1, col2= st.columns(2)
    with col1:
        if st.button( "future"):
            st.write("ok")
    with col2:
        if st.button( "deg3 now" if st.session_state.prdegree ==3 else "deg2 now"):
            if st.session_state.prdegree == 2:
                st.session_state.prdegree = 3
            else:
                st.session_state.prdegree = 2
            st.rerun()
            
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
    poly_model = make_pipeline(PolynomialFeatures(degree=st.session_state.prdegree), LinearRegression())
    poly_model.fit(time_steps, double_ma_data)

    # Create time steps for the future based on user selection
    future_time_steps = np.arange(len(double_ma_data), len(double_ma_data) + prediction_days).reshape(-1, 1)

    # Predict future values using both models
    linear_future_predictions = linear_model.predict(future_time_steps)
    poly_future_predictions = poly_model.predict(future_time_steps)

    # Get the date for the future prediction and last predicted value
    current_date = datetime.now(LOCAL_TIMEZONE)
    future_date = current_date + timedelta(days=prediction_days)
    future_date_str = future_date.strftime("%Y-%m-%d")
    last_linear_prediction = linear_future_predictions[-1]
    last_poly_prediction = poly_future_predictions[-1]
    
    # Plot the trend analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    
#######
    
    # Add the current date
    current_tinnitus_level = data[-1]  # Latest value in the tinnitus data
    current_double_ma_tinnitus_level = double_ma_data[-1]  # Latest value in the double moving average data
    
    # Display the current date and values in the plot
    today_date = current_date.strftime("%Y-%m-%d")
    km = distance
    ax.text(0.3, 0.85, f"慢跑第{days_difference}天,共{km}K\n\n{today_date} \n耳鸣级数：{current_tinnitus_level:.2f}\n双动态均值：{current_double_ma_tinnitus_level:.2f}", 
        horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, fontsize=12, color="blue")
  
    ax.scatter(time_steps, double_ma_data, color="blue", label="双动态均值（7日）")
    ax.plot(time_steps, linear_model.predict(time_steps), color="red", label="线性回归")
    ax.plot(time_steps, poly_model.predict(time_steps), color="green", label=f"多项式回归 ({st.session_state.prdegree}次)")
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
        color="blue",
        ha="center"
    )

    ax.annotate(
        f"{future_date_str}: {last_linear_prediction:.2f} (线性)",
        xy=(future_time_steps[-1], last_linear_prediction),
        xytext=(future_time_steps[-1], last_linear_prediction + 0.1),
        arrowprops=dict(facecolor='darkorange', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="darkorange",
        ha="center"
    )

    ax.annotate(
        f"{future_date_str}: {last_poly_prediction:.2f} (多项式)",
        xy=(future_time_steps[-1], last_poly_prediction),
        xytext=(future_time_steps[-1], last_poly_prediction + 0.1),
        arrowprops=dict(facecolor='purple', shrink=0.05, headwidth=10, width=2),
        fontsize=10,
        color="purple",
        ha="center"
    )

    # Add the last 28 points of data to the plot no no no
    last_21_days_data = data[-60:]
    last_21_days_time_steps = np.arange(len(double_ma_data) -60, len(double_ma_data)).reshape(-1, 1)

    ax.scatter(last_21_days_time_steps, last_21_days_data, color="gray", label="最近8周耳鸣级数", marker=".", s=100)
    ax.scatter(last_21_days_time_steps[-1], last_21_days_data[-1], color="Red", label="今天耳鸣级数",s=70)
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

##关联性分析

# Function to calculate double moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def double_moving_average(data, window_size):
    first_ma = moving_average(data, window_size)
    second_ma = moving_average(first_ma, window_size)
    return second_ma

def process_data (dataA, dataB):
    # Smooth user input
    window_size = 7

    user_data1_smooth = double_moving_average(dataA, window_size) 
    user_data2_smooth = double_moving_average(dataB, window_size) 

    dataNames = ["耳鸣","脾胃","睡眠","慢跑心率(最高值百分比/10)","慢跑5K时间(分钟/10)"]
    data_list = [data, data1, data2, data3, data4]
    nameA = ""
    nameB = ""
    for i in range(5):
        if dataA == data_list[i]:
            nameA = dataNames[i]
        if dataB == data_list[i]:
            nameB = dataNames[i]

    xiaimi_data_num = days_difference - 110 #(70 on 180 days of running)
    lengthHere = min(len(user_data1_smooth),len(user_data2_smooth))

   # xiaimi_data_num = lengthHere

    if dataA in (data3, data4) or dataB in (data3, data4) :
        user_data1_smooth = user_data1_smooth[-xiaimi_data_num:]
        user_data2_smooth = user_data2_smooth[-xiaimi_data_num:]

    return user_data1_smooth, user_data2_smooth, nameA, nameB

def single_correlation(dataA, dataB, num=0):
    user_data1_smooth, user_data2_smooth, nameA, nameB = process_data (dataA, dataB)
    # Scatter plot with different colors for data1 and data2
    scatter1 = axes[num].scatter(range(len(user_data1_smooth)), user_data1_smooth, color='blue')
    scatter2 = axes[num].scatter(range(len(user_data2_smooth)), user_data2_smooth, color='orange')

    ### original data last 30 days
    last_21_days_data1 = dataA[-30:] ### borrowed variable name
    last_21_days_time_steps = np.arange(len(user_data1_smooth) -30, len(user_data1_smooth)).reshape(-1, 1)
    axes[num].scatter(last_21_days_time_steps, last_21_days_data1, color="brown", label=f"最近30天原始数据", marker=".", s=60)

    last_21_days_data2 = dataB[-30:] ### borrowed variable name
    #last_21_days_time_steps = np.arange(len(user_data1_smooth) -60, len(user_data1_smooth)).reshape(-1, 1)
    axes[num].scatter(last_21_days_time_steps, last_21_days_data2, color="gray", marker=".", s=60)
    
    # Add trend lines and calculate R-squared values
    X = np.arange(len(user_data1_smooth)).reshape(-1, 1)
    
    # Linear regression for data1
    linear_model_data1 = LinearRegression()
    linear_model_data1.fit(X, user_data1_smooth)
    y_pred_linear_data1 = linear_model_data1.predict(X)
    r2_linear_data1 = r2_score(user_data1_smooth, y_pred_linear_data1)
    
    # Linear regression for data2
    linear_model_data2 = LinearRegression()
    linear_model_data2.fit(X, user_data2_smooth)
    y_pred_linear_data2 = linear_model_data2.predict(X)
    r2_linear_data2 = r2_score(user_data2_smooth, y_pred_linear_data2)
    
    # Polynomial regression for data1 (degree=2)
    poly_features = PolynomialFeatures(degree=st.session_state.prdegree)
    X_poly = poly_features.fit_transform(X)
    poly_model_data1 = LinearRegression()
    poly_model_data1.fit(X_poly, user_data1_smooth)
    y_pred_poly_data1 = poly_model_data1.predict(X_poly)
    r2_poly_data1 = r2_score(user_data1_smooth, y_pred_poly_data1)
    
    # Polynomial regression for data2 (degree=2)
    poly_model_data2 = LinearRegression()
    poly_model_data2.fit(X_poly, user_data2_smooth)
    y_pred_poly_data2 = poly_model_data2.predict(X_poly)
    r2_poly_data2 = r2_score(user_data2_smooth, y_pred_poly_data2)
    
    # Compare models and choose the better one
    if r2_poly_data1 > r2_linear_data1:
        best_model_data1 = poly_model_data1
        best_r2_data1 = r2_poly_data1
        model_type_data1 = "Polynomial"
    else:
        best_model_data1 = linear_model_data1
        best_r2_data1 = r2_linear_data1
        model_type_data1 = "Linear"
    
    if r2_poly_data2 > r2_linear_data2:
        best_model_data2 = poly_model_data2
        best_r2_data2 = r2_poly_data2
        model_type_data2 = "Polynomial"
    else:
        best_model_data2 = linear_model_data2
        best_r2_data2 = r2_linear_data2
        model_type_data2 = "Linear"
    
    # Plot trend lines
    axes[num].plot(X, y_pred_linear_data1, color='blue', linestyle='--', label=f"线性趋势 ({nameA}, $R^2$={r2_linear_data1:.2f})")
    axes[num].plot(X, y_pred_poly_data1, color='blue', linestyle=':', label=f"多项式趋势 ({nameA},{st.session_state.prdegree}次 $R^2$={r2_poly_data1:.2f})")
    
    axes[num].plot(X, y_pred_linear_data2, color='orange', linestyle='--', label=f"线性趋势 ({nameB}, $R^2$={r2_linear_data2:.2f})")
    axes[num].plot(X, y_pred_poly_data2, color='orange', linestyle=':', label=f"多项式趋势 ({nameB},{st.session_state.prdegree}次 $R^2$={r2_poly_data2:.2f})")
    
    # Predict trend for the next 30 days
    future_days = 30
    future_X = np.arange(len(user_data1_smooth), len(user_data1_smooth) + future_days).reshape(-1, 1)
    
    # Predict future values using the best model
    if model_type_data1 == "Polynomial":
        future_X_poly = poly_features.transform(future_X)
        future_data1 = best_model_data1.predict(future_X_poly)
    else:
        future_data1 = best_model_data1.predict(future_X)
    
    if model_type_data2 == "Polynomial":
        future_X_poly = poly_features.transform(future_X)
        future_data2 = best_model_data2.predict(future_X_poly)
    else:
        future_data2 = best_model_data2.predict(future_X)
    
    # Plot predicted trend
    axes[num].plot(range(len(user_data1_smooth), len(user_data1_smooth) + future_days), future_data1, color='blue', linestyle="-") # {model_type_data1})"
    axes[num].plot(range(len(user_data2_smooth), len(user_data2_smooth) + future_days), future_data2, color='orange', linestyle="-")#{model_type_data2})")
    axes[num].set_facecolor(bgColor)
    # Add a legend with a custom font size for all labels
    axes[num].legend(prop={'size': 5})  # Change '12' to your desired font size{num}
    axes[num].grid()
    # {num}Add date label and arrow for the last predicted point (30th day)
    chicago_tz = pytz.timezone("America/Chicago")
    last_date = datetime.now(chicago_tz) + timedelta(days=future_days)
    last_date_str = last_date.strftime("%m_%d")
    
    # Label and arrow for data1
    last_point_data1 = future_data1[-1]
    axes[num].annotate(
        f"{nameA}\n{last_date_str}\n{last_point_data1:.2f}",
        xy=(len(user_data1_smooth) + future_days - 1, last_point_data1),
        xytext=(len(user_data1_smooth) + future_days - 1, last_point_data1 ),  # Reduced height
        arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=5),
        fontsize=10,  # Smaller font size
        color='blue'
    )
    
    # Label and arrow for data2
    last_point_data2 = future_data2[-1]
    axes[num].annotate(
        f"{nameB}\n{last_date_str}\n{last_point_data2:.2f}",
        xy=(len(user_data2_smooth) + future_days - 1, last_point_data2),
        xytext=(len(user_data2_smooth) + future_days - 1, last_point_data2 ),  # Reduced height
        arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=5),
        fontsize=10,  # Smaller font size
        color='orange'
    )
        
    current_date = datetime.now(chicago_tz).strftime("%Y-%m-%d")
    km = distance
    if dataA == dataaa and dataB== databb:       
        axes[num].text(-3, -1,
            f"{current_date},第{days_difference}天,共{km}K：\n{dayData}\n{thought_input}",
            fontsize=15,  # Smaller font size
            color='navy',
            ha='left'
        )
    # Add date label in the bottom-right corner
   # st.markdown(f"Date: {current_date}")
    # Display correlation coefficient below the plot
    title = f"{current_date}慢跑第{days_difference}天,{nameA}和{nameB}双动态均值：相关性和趋势分析" 
    correlation_coeff = f"__线性相关系数:{np.corrcoef(user_data1_smooth, user_data2_smooth)[0, 1]:.2f}"
    #print(22, f"**Correlation Coefficient:** {np.corrcoef(user_data1_smooth, user_data2_smooth)[0, 1]:.2f}")
    axes[num].set_xlabel("天数")
    axes[num].set_ylabel("双动态均值")
    axes[num].set_title(title + correlation_coeff, fontsize = 15)
    axes[num].legend(loc='center left')

#############
#############-$# Plot correlation between smoothed data1 and data2
data_list = [data, data1, data2, data3, data4]
data_pairs = [(data, data1),(data, data2),(data1, data2),(data4, data3),(data, data3),(data, data4),(data1, data3),(data2, data3)]

##single corr plots
# Dropdown menu options
options = [
    "耳鸣 脾胃",
    "耳鸣 睡眠",
    "脾胃 睡眠",
    "慢跑心率 慢跑时长",
    "耳鸣 慢跑心率",
    "耳鸣 慢跑时长",
    "脾胃 慢跑心率",
    "睡眠 慢跑心率"
]


#st.image(imageName,width=705)
# Create a dropdown menu
st.header("相关度分析：单图")
            
selected_option = st.selectbox("Choose one correlation analysis:", options)

fig, axes = plt.subplots(2, 1, figsize=(12,9), gridspec_kw={'height_ratios': [9, 0.01]})
# Perform actions based on the selected option
dataaa, databb = data_pairs[0]
for i in range(8):
    if selected_option == options[i]:
        dataaa, databb = data_pairs[i]
        
single_correlation(dataaa, databb, 0)
st.pyplot(fig)


st.markdown("")
#st.image(imageName,width=705)
st.header("相关度分析：多图一组")

####### 3 & 5 plots together
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))  # 8 rows, 1 column
for i, (dataA, dataB) in enumerate(data_pairs[:3]):
    single_correlation(dataA, dataB, i)
plt.tight_layout()
st.pyplot(fig)

st.write("================")

