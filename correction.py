import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar

# Load data from CSV file
df = pd.read_csv('weather_data_kodaikanal_daily.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Check if the DataFrame is empty or required columns are missing
required_columns = ['temperature', 'dew_point', 'humidity', 'wind_speed', 'wind_gust', 'pressure', 'precipitation']
if df.empty or not all(col in df.columns for col in required_columns):
    st.error("Error: Required data columns are missing or data is empty.")
    st.stop()

# Simulate plant growth data
df['plant_growth'] = np.random.uniform(0, 10, len(df))

# Check for any missing values and handle them if necessary
df = df.fillna(method='ffill')

# Add simulated wind direction for demonstration
df['wind_direction'] = np.random.uniform(0, 360, len(df))

# Define features and target variable
features = ['temperature', 'dew_point', 'humidity', 'wind_speed', 'wind_gust', 'pressure', 'precipitation', 'wind_direction']
target = 'plant_growth'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title('🌱 Plant Growth Prediction Based on Weather Data on Hill Station 🌞')

# User inputs for new weather data
st.sidebar.header('🌦️ Input New Weather Data')
new_data = {
    'temperature': st.sidebar.number_input('🌡️ Temperature', min_value=0, max_value=100, value=27),
    'dew_point': st.sidebar.number_input('🌫️ Dew Point', min_value=0, max_value=30, value=5),
    'humidity': st.sidebar.number_input('💧 Humidity', min_value=0, max_value=100, value=60),
    'wind_speed': st.sidebar.number_input('🌬️ Wind Speed', min_value=0, max_value=20, value=5),
    'wind_gust': st.sidebar.number_input('🌪️ Wind Gust', min_value=0, max_value=30, value=10),
    'pressure': st.sidebar.number_input('🌡️ Pressure', min_value=950, max_value=1050, value=1000),
    'precipitation': st.sidebar.number_input('🌧️ Precipitation', min_value=0.0, max_value=50.0, value=5.0),
    'wind_direction': st.sidebar.number_input('🌬️ Wind Direction (°)', min_value=0, max_value=360, value=180)  # Adding wind direction
}

# Add a year selector to the sidebar
selected_year = st.sidebar.selectbox('📅 Select Year', options=df.index.year.unique(), index=len(df.index.year.unique()) - 1)
# Add a year selector to the sidebar with a fixed selection of 2028
#selected_year = st.sidebar.selectbox('📅 Select Year', options=[2028], index=0)

# Add a chart type selector to the sidebar
chart_type = st.sidebar.radio('🗺️ Select Chart Type', options=['2D', '3D'])

# Filter data based on the selected year
df_selected_year = df[df.index.year == selected_year]

# Convert the input to a DataFrame
new_data_df = pd.DataFrame([new_data])

# Make prediction
predicted_growth = model.predict(new_data_df)[0]

# Add the new data point to the dataframe
new_data_df['plant_growth'] = predicted_growth
new_data_df.index = [df.index.max() + pd.Timedelta(days=1)]
df = pd.concat([df, new_data_df])

# Display the historical data for the selected year
st.subheader(f'📈 Historical Plant Growth Data for {selected_year}')
fig_hist = make_subplots(specs=[[{"secondary_y": True}]])

# Line plot for Temperature
fig_hist.add_trace(go.Scatter(
    x=df_selected_year.index,
    y=df_selected_year['temperature'],
    mode='lines',
    name='Temperature',
    line=dict(color='blue')),
    secondary_y=False
)

# Line plot for Plant Growth
fig_hist.add_trace(go.Scatter(
    x=df_selected_year.index,
    y=df_selected_year['plant_growth'],
    mode='lines+markers',
    name='Plant Growth',
    line=dict(color='green')),
    secondary_y=True
)

fig_hist.update_layout(
    title=f'Historical Plant Growth and Temperature for {selected_year} 📈',
    yaxis_title='Temperature (°C)',
    yaxis2_title='Plant Growth'
)
st.plotly_chart(fig_hist)

# Display the prediction
st.subheader(f"### 🌱 Predicted Plant Growth: **{predicted_growth:.2f}**")

# Display the prediction graph
st.subheader('🌟 Predicted Plant Growth 🌟')
if chart_type == '2D':
    fig_pred = px.line(df, x=df.index, y='plant_growth', title='Plant Growth Prediction 🔮')
    st.plotly_chart(fig_pred)
elif chart_type == '3D':
    fig_pred = go.Figure(data=[go.Scatter3d(
        x=df.index,
        y=df['temperature'],
        z=df['plant_growth'],
        mode='markers',
        marker=dict(size=5, color=df['plant_growth'], colorscale='Viridis', showscale=True)
    )])
    fig_pred.update_layout(title='3D Plant Growth Prediction 🔮', scene=dict(xaxis_title='Date', yaxis_title='Temperature', zaxis_title='Plant Growth'))
    st.plotly_chart(fig_pred)

# Bubble Chart
st.subheader('🔵 Temperature')
if chart_type == '2D':
    fig_bubble = px.scatter(df, x='temperature', y='plant_growth', size='humidity', color='dew_point', title='Bubble Chart')
    st.plotly_chart(fig_bubble)
elif chart_type == '3D':
    fig_bubble = go.Figure(data=[go.Scatter3d(
        x=df['temperature'],
        y=df['plant_growth'],
        z=df['humidity'],
        mode='markers',
        marker=dict(size=5, color=df['dew_point'], colorscale='Viridis', showscale=True)
    )])
    fig_bubble.update_layout(title='3D Bubble Chart', scene=dict(xaxis_title='Temperature', yaxis_title='Plant Growth', zaxis_title='Humidity'))
    st.plotly_chart(fig_bubble)

# Gauge Chart
st.subheader('🎯 Growth Percentage')
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=predicted_growth,
    title={'text': "Predicted Plant Growth"},
    gauge={'axis': {'range': [0, 10]}}
))
st.plotly_chart(fig_gauge)

# Area Chart
st.subheader('🌊 Plant Yield')
if chart_type == '2D':
    fig_area = px.area(df, x=df.index, y='plant_growth', title='Area Chart')
    st.plotly_chart(fig_area)
elif chart_type == '3D':
    fig_area = go.Figure(data=[go.Scatter3d(
        x=df.index,
        y=df['plant_growth'],
        z=df['temperature'],
        mode='lines',
        line=dict(color='blue')
    )])
    fig_area.update_layout(title='3D Area Chart', scene=dict(xaxis_title='Date', yaxis_title='Plant Growth', zaxis_title='Temperature'))
    st.plotly_chart(fig_area)

# Mekko Chart (using a simple stacked bar chart as a substitute)
st.subheader('📊 MONTH ANALYSIS')
if chart_type == '2D':
    mekko_data = df.groupby(pd.Grouper(freq='M')).mean()
    fig_mekko = px.bar(mekko_data, x=mekko_data.index, y='plant_growth', title='Health of Tree', text_auto=True)
    st.plotly_chart(fig_mekko)
elif chart_type == '3D':
    mekko_data = df.groupby(pd.Grouper(freq='M')).mean()
    fig_mekko = go.Figure(data=[go.Bar(
        x=mekko_data.index,
        y=mekko_data['plant_growth'],
        marker=dict(color='green')
    )])
    fig_mekko.update_layout(title='3D Month Analysis', xaxis_title='Date', yaxis_title='Plant Growth')
    st.plotly_chart(fig_mekko)

# Cone Chart (Reflecting Wind Direction with Compass Directions)
st.subheader('🧭 WIND DIRECTION')
fig_cone = go.Figure()

# Cone representation of wind direction
fig_cone.add_trace(go.Cone(
    x=[0], y=[0], z=[0],
    u=[new_data['wind_speed'] * np.cos(np.radians(new_data['wind_direction']))],
    v=[new_data['wind_speed'] * np.sin(np.radians(new_data['wind_direction']))],
    w=[0],
    anchor='tail',
    showscale=False
))

# Add compass direction annotations
compass_directions = {
    'N': (0, 1.1),
    'S': (0, -1.1),
    'E': (1.1, 0),
    'W': (-1.1, 0)
}
for direction, (x, y) in compass_directions.items():
    fig_cone.add_annotation(
        x=x,
        y=y,
        text=direction,
        showarrow=False,
        font=dict(size=14, color="black")
    )

fig_cone.update_layout(
    title='Wind Direction with Compass',
    xaxis=dict(range=[-2, 2], title='X'),
    yaxis=dict(range=[-2, 2], title='Y'),
    showlegend=False
)
st.plotly_chart(fig_cone)

# Highlight the user inputs
st.sidebar.write("### 🌦️ Input Summary")
st.sidebar.write(f"**Temperature:** {new_data['temperature']} °C")
st.sidebar.write(f"**Dew Point:** {new_data['dew_point']} °C")
st.sidebar.write(f"**Humidity:** {new_data['humidity']} %")
st.sidebar.write(f"**Wind Speed:** {new_data['wind_speed']} km/h")
st.sidebar.write(f"**Wind Gust:** {new_data['wind_gust']} km/h")
st.sidebar.write(f"**Pressure:** {new_data['pressure']} hPa")
st.sidebar.write(f"**Precipitation:** {new_data['precipitation']} mm")
st.sidebar.write(f"**Wind Direction:** {new_data['wind_direction']} °")
