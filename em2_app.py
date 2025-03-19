#em2 app 3-12-25
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import pytz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.font_manager as fm
import textwrap
from sklearn.metrics import r2_score
import os

from textblob import TextBlob

CSV_FILE = "mood_history.csv"
MOOD_EMOJIS = {
    'Happy': '😊', 
    'Sad': '😢',
    'Energetic': '💪',
    'Calm': '🧘',
    'Creative': '🎨',
    'Stressed': '😰',
    'Neutral': '😐'
}

def init_session_state():
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = pd.DataFrame(columns=[
            "Date", "Sentence 1", "Sentence 2", 
            "Predicted Mood", "Mood Score", "Emoji"
        ])

#st.set_page_config(page_title="Mood Diary", page_icon="📔")
st.title("📔 Daily Mood Diary")
st.write("Document your daily mood with two sentences!")

# Allow user to upload a CSV file
uploaded_file = st.file_uploader("Upload your mood history CSV", type=["csv"])
if uploaded_file is not None:
    st.session_state.mood_history = pd.read_csv(uploaded_file, parse_dates=["Date"])

init_session_state()

st.write("## Daily Entry")

# Determine if today's entry exists
today_date = datetime.now().date()
submitted_today = False
if not st.session_state.mood_history.empty and 'Date' in st.session_state.mood_history:
    submitted_today = today_date in pd.to_datetime(st.session_state.mood_history["Date"]).dt.date.values

with st.form("daily_entry"):
    sentence1 = st.text_area("First sentence:", disabled=submitted_today,
                             placeholder="How are you feeling today?", height=68)
    sentence2 = st.text_area("Second sentence:", disabled=submitted_today,
                             placeholder="What's been on your mind?", height=68)
    submitted = st.form_submit_button("Save Today's Entry", disabled=submitted_today)

def analyze_mood(sentence1, sentence2):
    combined_text = f"{sentence1} {sentence2}"
    analysis = TextBlob(combined_text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        mood = "Happy"
    elif polarity < -0.3:
        mood = "Sad"
    else:
        mood = "Neutral"
        
    return mood, polarity, MOOD_EMOJIS.get(mood, "")

if submitted and sentence1.strip() and sentence2.strip():
    mood, score, emoji = analyze_mood(sentence1, sentence2)
    today = datetime.now()
    
    new_entry = {
        "Date": today,
        "Sentence 1": sentence1,
        "Sentence 2": sentence2,
        "Predicted Mood": mood,
        "Mood Score": round(score, 2),
        "Emoji": emoji
    }
    
    new_entry_df = pd.DataFrame([new_entry])
    st.session_state.mood_history = pd.concat([st.session_state.mood_history, new_entry_df], ignore_index=True)
    
    # Save to server CSV only if no file was uploaded
    if uploaded_file is None:
        new_entry_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    
    st.success("Entry saved successfully!")
    st.balloons()
    submitted_today = True  # Immediately reflect the submission

elif submitted:
    st.warning("Please fill in both sentences!")

if submitted_today:
    st.subheader("Today's Mood")
    col1, col2, col3 = st.columns(3)
    today_entry = st.session_state.mood_history.iloc[-1]
    with col1:
        st.metric("Predicted Mood", f"{today_entry['Predicted Mood']} {today_entry['Emoji']}")
    with col2:
        st.metric("Mood Score", f"{today_entry['Mood Score']:.2f}")
    with col3:
        csv = st.session_state.mood_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History",
            data=csv,
            file_name="mood_history.csv",
            mime="text/csv"
        )

if not st.session_state.mood_history.empty:
    st.subheader("Your Mood History")
    display_df = st.session_state.mood_history.tail(5).copy()
    display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.style.format({"Mood Score": "{:.2f}"}))

    st.subheader("Mood Timeline (Current Month)")
    current_month = datetime.now().month
    current_year = datetime.now().year

    monthly_data = st.session_state.mood_history[
        (pd.to_datetime(st.session_state.mood_history["Date"]).dt.month == current_month) &
        (pd.to_datetime(st.session_state.mood_history["Date"]).dt.year == current_year)]
    
    if not monthly_data.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(monthly_data["Date"]), monthly_data["Mood Score"], 
                marker='o', linestyle='-', color='skyblue', label="Mood Score")
        
        for date, score, emoji in zip(pd.to_datetime(monthly_data["Date"]), monthly_data["Mood Score"], monthly_data["Emoji"]):
            ax.text(date, score + 0.02, emoji, fontsize=12, ha='center', va='bottom', color="orange")
        
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d'))
        plt.xticks(rotation=45)
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Mood Score")
        ax.set_title(f"Mood Timeline ({datetime.now().strftime('%B %Y')})")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No data available for the current month.")

st.markdown("---")
st.caption("Your personal mood diary - Reflect, remember, and grow.")
##################################

#divider image
imageName = "cherry.jpeg"
st.image(imageName,width=705)
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

set_message="健康第一，做有意义的事，不懈怠，不贪不嗔不痴，无证据不假定"
# Calculate updated Days and Distance based on the current date
LOCAL_TIMEZONE = pytz.timezone('America/Chicago')  # Replace with your local timezone
current_time = datetime.now(LOCAL_TIMEZONE)

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

st.image(imageName,width=710)
##############。 耳鸣分析
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


# Parse input data
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
with st.sidebar:
    #st.header("Data Upload")
    # Add checkbox to toggle upload visibility
    show_upload = st.checkbox("Upload local health data?", value=False)
    
    if show_upload:
        uploaded_file = st.file_uploader("Choose txt file", type=["txt"])
        
        if uploaded_file is not None:

            with open(uploaded_file, "r") as file:
                uploadFile = file.read()
                st.sidebar.code(uploadFile)
                
#st.sidebar.text_area("复制以下格式化字符串：", value=formatted_string, height=100, key="formatted_output")
with open('rebootdate.txt', "r") as file:
    reboot = file.read()

st.sidebar.code(formatted_string, language="plaintext")
st.sidebar.code(reboot,language="plaintext")

# Add a "key" input box for automatic saving
st.sidebar.subheader("自动保存设置")
key_input = st.sidebar.text_input("输入密钥以自动保存数据:", type="password")  # Hide the entered key

# Automatically save data if the key is "zzzzzzzzz"
if key_input.strip() == "zzzzzzzzz":
    if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
        save_data(data, data1, data2, dataa, datab)
        st.sidebar.success("数据已自动保存！")
    else:
        st.sidebar.error("无法保存数据，请检查输入格式。")

# Add a download button for the formatted string
    st.sidebar.download_button(
        label="下载为文本文件",
        data=formatted_string,
        file_name="健康数据.txt",
        mime="text/plain"
    )

# Add a button to download the saved data as a CSV file
#if st.sidebar.button("下载数据为CSV文件"):
    #if data is not None and data1 is not None and data2 is not None and data3 is not None and data4 is not None:
       # save_data(data, data1, data2, dataa, datab, "health_data.csv")
        #with open("health_data.csv", "rb") as file:
         #   st.sidebar.download_button(
             #   label="点击下载CSV文件",
               # data=file,
               # file_name="health_data.csv",
               # mime="text/csv"
            #)
    #else:
       # st.sidebar.error("没有可下载的数据，请先输入数据并保存。")

# Add a divider before the new section
st.sidebar.markdown("---")

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
    
    #st.sidebar.subheader("marker_message")
    marker_message_input = st.sidebar.text_area("输入分析信息：", value="")
    # Function to wrap text into lines of a specified width
    def wrap_text(text, width):
        return "\n".join(textwrap.wrap(text, width=width))

# Position and content for the text annotation
    
    current_date = datetime.now(LOCAL_TIMEZONE).strftime("%Y年%m月%d日")
    message1 = f"慢跑第{days_difference}天, {current_date}: \n"
    wrapped_input = wrap_text(marker_message_input, 25)  # Wrap text to 25 characters per line
    dayData = f"耳鸣：{data[-1]}, 脾胃：{data1[-1]}, 睡眠：{data2[-1]},慢跑心率（最高值百分比）：{data3[-1] *10}, 时长：{data4[-1]*10},"
    
    ax.text(v_line_pos, 3.5,
        message1 +set_message   + "\n"+ wrapped_input + "\n"+ dayData,
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
    
#######
    
    # Add the current date
    current_tinnitus_level = data[-1]  # Latest value in the tinnitus data
    current_double_ma_tinnitus_level = double_ma_data[-1]  # Latest value in the double moving average data
    
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
    poly_features = PolynomialFeatures(degree=2)
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
    axes[num].plot(X, y_pred_poly_data1, color='blue', linestyle=':', label=f"多项式趋势 ({nameA}, $R^2$={r2_poly_data1:.2f})")
    
    axes[num].plot(X, y_pred_linear_data2, color='orange', linestyle='--', label=f"线性趋势 ({nameB}, $R^2$={r2_linear_data2:.2f})")
    axes[num].plot(X, y_pred_poly_data2, color='orange', linestyle=':', label=f"多项式趋势 ({nameB}, $R^2$={r2_poly_data2:.2f})")
    
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
    
    # Add date label in the bottom-right corner
    current_date = datetime.now(chicago_tz).strftime("%Y-%m-%d")
    
   # st.markdown(f"Date: {current_date}")
    # Display correlation coefficient below the plot
    title = f"{current_date}{nameA}和{nameB}双动态均值：相关性和趋势分析" 
    correlation_coeff = f"__相关系数:{np.corrcoef(user_data1_smooth, user_data2_smooth)[0, 1]:.2f}"
    #print(22, f"**Correlation Coefficient:** {np.corrcoef(user_data1_smooth, user_data2_smooth)[0, 1]:.2f}")
    
    axes[num].set_xlabel("天数")
    axes[num].set_ylabel("双动态均值")
    axes[num].set_title(title + correlation_coeff)
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


st.image(imageName,width=705)
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
st.image(imageName,width=705)
st.header("相关度分析：多图一组")

####### 3 & 5 plots together
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))  # 8 rows, 1 column
for i, (dataA, dataB) in enumerate(data_pairs[:3]):
    single_correlation(dataA, dataB, i)
plt.tight_layout()
st.pyplot(fig)

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 24))  # 8 rows, 1 column
for i, (dataA, dataB) in enumerate(data_pairs[3:]):
    single_correlation(dataA, dataB, i)
plt.tight_layout()
st.pyplot(fig)

#######################

# Option 1: Using a local image file
# Place the image in the same directory as your script or provide the correct path
image_url = "mycloud.jpeg"  # Replace with your local image file name
st.image(image_url, caption="This is a local image", width=705)

#video
local_video_path = "taiji.mp4"  # Replace with your local video file name
# Display the 
st.video(local_video_path)


###############
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os

# Set up the Streamlit app title
st.title( "Celebration of life, growth, and enlightenment")

# Define the shapes of letters and numbers with improved coordinates
def get_letter_shapes():
    # Coordinates for the characters (scaled and centered)
    H = np.array([
        [-0.8, -1], [-0.8, 1],  # Left vertical
        [0.8, -1], [0.8, 1],    # Right vertical
        [-0.8, 0], [0.8, 0]     # Horizontal bar
    ]) * 1.5
    
    Six = np.array([
        [1, 1],          # Top start
        [-1, 0.5],       # Left curve top
        [-1, -0.5],      # Left curve bottom
        [0, -1],         # Bottom center
        [1, -0.5],       # Right curve bottom
        [0.5, 0],        # Inner curve
        [0, 0.5]         # Closing point
    ]) * 1.5
    
    Zero = np.array([
        [0, 1], [-0.5, 0.87], [-0.87, 0.5], [-1, 0],  # Left semicircle
        [-0.87, -0.5], [-0.5, -0.87], [0, -1],        # Bottom semicircle
        [0.5, -0.87], [0.87, -0.5], [1, 0],           # Right semicircle
        [0.87, 0.5], [0.5, 0.87], [0, 1]              # Top semicircle
    ]) * 1.5

    # Points for "Y"
    Y = np.array([[-1, 1], [0, 0], [0, -1], [0.25, 0], [1, 1]]) * 2

    W = np.array([[-1, 0.75], [-0.5, -0.75], [0, 0], [0.5, -0.75], [1, 0.75]]) * 2

    # Edges for each character (explicit connections between points)
    letter_edges = {
        'H': [(0, 1), (2, 3), (4, 5)],  # Connect left, right, and middle bars
        'A': [(0, 1), (1, 2), (2, 3), (4, 5)],  # Triangle and horizontal bar
        'P': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],  # Circular P
        'Y': [(0, 1), (1, 2), (2, 3), (2, 4)],  # Correct "Y" shape
        'W': [(0, 1), (1, 2), (2, 3), (3, 4)],  # Peaks of M
        'S': [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0)],  # Continuous 6 shape
        'Z': [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,0)],  # Full circle
    }

    return {
        'H': H, 
        'A': np.array([[-1, -1], [-0.5, 1], [0.5, 1], [1, -1], [-0.5, 0], [0.5, 0]]) * 2,
        'P': np.array([[-1, -1], [-1, 1], [0, 1], [1, 0], [-1, 0]]) * 2,
        'Y': Y,  # Updated "Y" points
        'W': W,
        'S': Six,
        'Z': Zero
    }, letter_edges

# Modified firework generation with better particle control
def generate_firework(ax, x_center, y_center, is_word=False, character=None):
    letter_shapes, letter_edges = get_letter_shapes()
    if is_word and character:
        num_particles_per_point = 15  # Increased particle density
        letter_shape = letter_shapes[character]
        edges = letter_edges[character]
        x, y = [], []
        
        # Add intermediate points for better shape definition
        for edge in edges:
            start_idx, end_idx = edge
            start = letter_shape[start_idx]
            end = letter_shape[end_idx]
            points = np.linspace(start, end, 5)
            for point in points:
                offsets = np.random.normal(loc=point, scale=0.1, size=(num_particles_per_point, 2))
                x.extend(offsets[:, 0] + x_center)
                y.extend(offsets[:, 1] + y_center)
        
        scatter = ax.scatter(x, y, s=20, c=np.random.rand(3,), alpha=0.95, edgecolors="none")
    else:
        num_particles = 200
        num_center_particles = int(num_particles * 0.2)
        num_outer_particles = num_particles - num_center_particles
        center_x = np.random.normal(loc=x_center, scale=0.5, size=num_center_particles)
        center_y = np.random.normal(loc=y_center, scale=0.5, size=num_center_particles)
        angles = np.linspace(0, 2 * np.pi, num_outer_particles)
        radii = np.random.uniform(1, 4, size=num_outer_particles)
        outer_x = x_center + radii * np.cos(angles)
        outer_y = y_center + radii * np.sin(angles)
        x = np.concatenate([center_x, outer_x])
        y = np.concatenate([center_y, outer_y])
        scatter = ax.scatter(x, y, s=15, c=np.random.rand(3,), alpha=0.9, edgecolors="none")
    return scatter

# Function to update the fireworks animation
def update(frame, ascending_fireworks, exploded_scatters, ax, series_launched):
    if frame == 50 and not series_launched[0]:
        series_letters = ['H','A','P','P','Y','H','W','S','Z']
        num_letters = len(series_letters)
        
        # Adjusted explosion heights with steeper progression
        base_height = 20
        explosion_heights = np.linspace(base_height, base_height + num_letters* 1.5, num_letters)
        
        # Space characters evenly with wider spread
        series_x = np.linspace(-18, 18, num_letters)
        
        # Create ascending fireworks for each character
        for i, char in enumerate(series_letters):
            scatter = ax.scatter(series_x[i], -5, s=10, c="white", alpha=0.8)
            ascending_fireworks.append({
                "scatter": scatter,
                "x": series_x[i],
                "y": -5,
                "speed": 1.2,  # Faster ascent for clearer sequence
                "explosion_height": explosion_heights[i],
                "character": char
            })
        series_launched[0] = True

    # Update ascending fireworks
    if ascending_fireworks:
        for firework in ascending_fireworks.copy():
            firework["y"] += firework["speed"]
            firework["scatter"].set_offsets([firework["x"], firework["y"]])
            if firework["y"] >= firework["explosion_height"]:
                if "character" in firework:
                    exploded_scatters.append(generate_firework(ax, firework["x"], firework["y"], is_word=True, character=firework["character"]))
                else:
                    if np.random.rand() < 0.3:
                        exploded_scatters.append(generate_firework(ax, firework["x"], firework["y"], is_word=True))
                    else:
                        exploded_scatters.append(generate_firework(ax, firework["x"], firework["y"]))
                firework["scatter"].remove()
                ascending_fireworks.remove(firework)

    # Update exploded fireworks particles
    if exploded_scatters:
        for scatter in exploded_scatters.copy():
            offsets = scatter.get_offsets()
            offsets[:, 1] -= 0.2
            scatter.set_offsets(offsets)
            current_alpha = scatter.get_alpha()
            new_alpha = max(0.0, min(1.0, current_alpha - 0.02))
            scatter.set_alpha(new_alpha)
            if new_alpha <= 0:
                scatter.remove()
                exploded_scatters.remove(scatter)

    # Add new ascending fireworks occasionally
    if frame % 10 == 0:
        x_start = np.random.uniform(-10, 10)
        y_start = -5
        explosion_height = np.random.uniform(15, 40)
        speed = np.random.uniform(0.5, 1.0)
        scatter = ax.scatter(x_start, y_start, s=10, c="white", alpha=0.8)
        ascending_fireworks.append({
            "scatter": scatter,
            "x": x_start,
            "y": y_start,
            "speed": speed,
            "explosion_height": explosion_height
        })

def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-10, 40)
    ax.axis("off")

    ascending_fireworks = []
    exploded_scatters = []
    series_launched = [False]  # Track if series has been launched

    ani = FuncAnimation(fig, update, frames=200, fargs=(ascending_fireworks, exploded_scatters, ax, series_launched), interval=50, blit=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
        try:
            ani.save(tmpfile.name, writer="pillow", fps=20)
            gif_path = tmpfile.name
        except Exception as e:
            st.error(f"Failed to save animation: {e}")
            return

    if st.button("Play Fireworks"):
        st.write("Playing fireworks...")
        st.image(gif_path, use_container_width=True)
        time.sleep(40)

    if os.path.exists(gif_path):
        os.unlink(gif_path)

if __name__ == "__main__":
    main()
