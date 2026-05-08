import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

st.set_page_config(
    page_title="Bike Rental Demand Dashboard",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🚲 Bike Rental Demand Prediction Dashboard")
st.markdown("Predict bike demand and analyze historical rental trends")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------------
# LOAD DATASET
# -----------------------------------
df = pd.read_csv("Dataset.csv")

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------
st.sidebar.header("📥 Input Parameters")

temp = st.sidebar.number_input(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

atemp = st.sidebar.number_input(
    "Feeling Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

hum = st.sidebar.number_input(
    "Humidity",
    min_value=0.0,
    max_value=1.0,
    value=0.50
)

windspeed = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

hr = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)

date = st.sidebar.date_input(
    "Select Date",
    datetime.date.today()
)

season = st.sidebar.selectbox(
    "Season",
    ["springer", "summer", "winter", "fall"]
)

holiday = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)

workingday = st.sidebar.selectbox(
    "Working Day",
    ["Yes", "No"]
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Mist", "Light Snow", "Heavy Rain"]
)

# -----------------------------------
# PREDICT BUTTON
# -----------------------------------
if st.sidebar.button("Predict Demand"):
    columns = [
        'hr','weekday','temp','atemp','hum','windspeed',
        'year','month','day','dayofweek','is_weekend',
        'season_springer','season_summer','season_winter',
        'yr_2012','mnth_10','mnth_11','mnth_12','mnth_2',
        'mnth_3','mnth_4','mnth_5','mnth_6','mnth_7',
        'mnth_8','mnth_9','holiday_Yes',
        'workingday_Working Day',
        'weathersit_Heavy Rain',
        'weathersit_Light Snow',
        'weathersit_Mist'
    ]

    input_df = pd.DataFrame(
        np.zeros((1, len(columns))),
        columns=columns
    )

    input_df['temp'] = temp
    input_df['atemp'] = atemp
    input_df['hum'] = hum
    input_df['windspeed'] = windspeed
    input_df['hr'] = hr

    input_df['day'] = date.day
    input_df['month'] = date.month
    input_df['year'] = date.year
    input_df['dayofweek'] = date.weekday()
    input_df['weekday'] = date.weekday()

    # Weekend
    if date.weekday() >= 5:
        input_df['is_weekend'] = 1

    if season == "springer":
        input_df['season_springer'] = 1

    elif season == "summer":
        input_df['season_summer'] = 1

    elif season == "winter":
        input_df['season_winter'] = 1

    if date.year == 2012:
        input_df['yr_2012'] = 1

    month_col = f"mnth_{date.month}"

    if month_col in input_df.columns:
        input_df[month_col] = 1

    # -----------------------------------
    # HOLIDAY
    # -----------------------------------
    if holiday == "Yes":
        input_df['holiday_Yes'] = 1

    # -----------------------------------
    # WORKING DAY
    # -----------------------------------
    if workingday == "Yes":
        input_df['workingday_Working Day'] = 1

    # -----------------------------------
    # WEATHER ENCODING
    # -----------------------------------
    if weather == "Mist":
        input_df['weathersit_Mist'] = 1

    elif weather == "Light Snow":
        input_df['weathersit_Light Snow'] = 1

    elif weather == "Heavy Rain":
        input_df['weathersit_Heavy Rain'] = 1

    # -----------------------------------
    # PREDICTION
    # -----------------------------------
    prediction = model.predict(input_df)

    demand = int(prediction[0])

    # -----------------------------------
    # KPI CARDS
    # -----------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🚲 Predicted Demand",
            value=demand
        )

    with col2:
        st.metric(
            label="🌡 Temperature",
            value=temp
        )

    with col3:
        st.metric(
            label="💧 Humidity",
            value=hum
        )

    # -----------------------------------
    # TABS
    # -----------------------------------
    tab1, tab2 = st.tabs(
        ["📊 Prediction", "📈 Analytics Dashboard"]
    )

    # ===================================
    # TAB 1 : PREDICTION
    # ===================================
    with tab1:

        st.subheader("📊 Prediction Result")

        st.success(
            f"🚀 Predicted Bike Demand: {demand}"
        )

        # Progress Bar
        st.progress(min(demand / 1000, 1.0))

        # Demand Insights
        if demand < 100:
            st.warning("⚠ Low demand expected")

        elif demand < 300:
            st.info("ℹ Moderate demand expected")

        else:
            st.success("🔥 High demand expected!")

    # ===================================
    # TAB 2 : ANALYTICS
    # ===================================
    with tab2:

        st.subheader("📈 Interactive Analytics Dashboard")

        # -----------------------------------
        # 1. Hourly Demand Trend
        # -----------------------------------
        st.markdown("### 🕒 Hourly Bike Demand")

        hourly_demand = (
            df.groupby('hr')['cnt']
            .mean()
            .reset_index()
        )

        fig1 = px.line(
            hourly_demand,
            x='hr',
            y='cnt',
            markers=True,
            title='Average Bike Demand by Hour'
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # -----------------------------------
        # 2. Season-wise Demand
        # -----------------------------------
        st.markdown("### 🌤 Season-wise Demand")

        season_demand = (
            df.groupby('season')['cnt']
            .mean()
            .reset_index()
        )

        fig2 = px.bar(
            season_demand,
            x='season',
            y='cnt',
            title='Average Demand by Season',
            text_auto=True
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # -----------------------------------
        # 3. Weather Impact
        # -----------------------------------
        st.markdown("### 🌧 Weather Impact")

        weather_demand = (
            df.groupby('weathersit')['cnt']
            .mean()
            .reset_index()
        )

        fig3 = px.bar(
            weather_demand,
            x='weathersit',
            y='cnt',
            title='Weather vs Bike Demand',
            text_auto=True
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # -----------------------------------
        # 4. Working Day Analysis
        # -----------------------------------
        st.markdown("### 🏢 Working Day Analysis")

        workingday_demand = (
            df.groupby('workingday')['cnt']
            .mean()
            .reset_index()
        )

        fig4 = px.pie(
            workingday_demand,
            names='workingday',
            values='cnt',
            title='Working Day Demand Distribution'
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )
