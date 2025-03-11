import pandas as pd
import numpy as np

# Generate date range from 2025 to 2030
date_range = pd.date_range(start="2025-01-01", end="2030-12-31", freq='D')

# Generate random weather data
np.random.seed(42)
temperature = np.random.uniform(5, 30, len(date_range))
dew_point = temperature - np.random.uniform(1, 5, len(date_range))
humidity = np.random.uniform(50, 100, len(date_range))
wind_speed = np.random.uniform(0, 15, len(date_range))
wind_gust = wind_speed + np.random.uniform(0, 5, len(date_range))
pressure = np.random.uniform(980, 1050, len(date_range))
precipitation = np.random.uniform(0, 10, len(date_range))

# Create DataFrame
weather_df = pd.DataFrame({
    "date": date_range,
    "temperature": temperature,
    "dew_point": dew_point,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "wind_gust": wind_gust,
    "pressure": pressure,
    "precipitation": precipitation
})

# Save to CSV
file_path = "/mnt/data/dummy_weather_data_2025_2030.csv"
weather_df.to_csv(file_path, index=False)

file_path
