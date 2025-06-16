
#%%
#import prophet
from prophet import Prophet
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

'''
prophet: 1.1.7 
pandas: 1.5.2 
matplotlib: 3.6.2 
numpy : 1.26.4
'''
'''`
print(
    "prophet:",prophet.__version__,'\n'
    , "pandas:",pd.__version__,'\n'
    , "matplotlib:",plt.__version__,'\n'
    , "numpy :",np.__version__
)
'''


train_path   = "data/weather_train.parquet"

# 1b) Load the training data
train = pd.read_parquet(train_path)

# ----test
import os, pathlib
p = pathlib.Path("data/weather_train.parquet")
print(p.stat().st_size, "bytes")   # should be > 0

import duckdb
duckdb.query("SELECT COUNT(*) FROM 'data/weather_train.parquet'").show()


# 1c) Select a single location for simplicity
city = "Chattanooga"  # change to your city

df_train = (
    train[train["location"] == city]
         .sort_values("date")
         .reset_index(drop=True)
)

print(f"Loaded {df_train.shape[0]} rows for {city}")
df_train.head()

# Ensure 'date' is a datetime (place at the top of ## 2)
if not pd.api.types.is_datetime64_any_dtype(df_train["date"]):
    df_train["date"] = pd.to_datetime(df_train["date"])

# Prophet expects columns 'ds' (date) and 'y' (value to forecast)
prophet_df = (
    df_train[["date", "t2m_max"]]
    .rename(columns={"date": "ds", "t2m_max": "y"})
)
prophet_df.head()

print("\n--- Prophet DataFrame Info ---")
prophet_df.info()
print("\n--- Duplicate Timestamps Check ---")
print(prophet_df["ds"].duplicated().sum())

# 3a) Instantiate Prophet with default seasonality
m = prophet.Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# 3b) Fit to the historical data
m.fit(prophet_df)

# 4a) Create a future dataframe extending 730 days (≈2 years), including history
future = m.make_future_dataframe(periods=365*2, freq="D")

# 4b) Generate the forecast once (contains both in-sample and future)
df_forecast = m.predict(future)

# 4c) Inspect the in-sample head and forecast tail:
print("-- In-sample --")
df_forecast[ ["ds", "yhat", "yhat_lower", "yhat_upper"] ].head()

#print("-- Forecast (2-year) --")
#df_forecast[ ["ds", "yhat", "yhat_lower", "yhat_upper"] ].tail()

# plot
# fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(14, 8))
ax.fill_between(pd.to_datetime(df_forecast['ds']), df_forecast['yhat_lower'], df_forecast['yhat_upper'], alpha=.5, linewidth=0)
ax.plot(pd.to_datetime(df_forecast['ds']), df_forecast['yhat'], linewidth=2)
plt.show()



# %%
