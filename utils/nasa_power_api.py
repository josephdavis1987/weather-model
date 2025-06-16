import pandas as pd
import requests

def fetch_nasa_power_data(
    lat: float,
    lon: float,
    location: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Fetch daily NASA POWER data for a single point (lat, lon)
    between `start` and `end` (YYYYMMDD strings).

    Returns a DataFrame with columns:
      date, location, lat, lon,
      t2m, t2m_max, t2m_min, t2m_range,
      t2m_dew (dew‐point),
      rh2m,     (relative humidity)
      ws10m,    (wind speed 10 m)
      wd10m,    (wind direction 10 m)
      ps,       (surface pressure)
      allsky_sw_dwn,  (shortwave down)
      allsky_lw_dwn,  (longwave down)
      prectotcorr,    (precipitation)
      day_of_year     (1–365/366)
    """
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
        "community": "AG",
        "parameters": ",".join([
            "T2M",             # daily mean temperature at 2 m
            "T2M_MAX",         # daily maximum temperature
            "T2M_MIN",         # daily minimum temperature
            "T2MDEW",          # daily dew-point temperature
            "RH2M",            # daily relative humidity
            "WS10M",           # daily wind speed at 10 m
            "WD10M",           # daily wind direction at 10 m
            "PS",              # daily surface pressure
            "ALLSKY_SFC_SW_DWN",  # daily all-sky shortwave down
            "ALLSKY_SFC_LW_DWN",  # daily all-sky longwave down
            "PRECTOTCORR"      # daily corrected precipitation
        ])
    }

    # 1) Make the API request
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # 2) Drill into the JSON block that holds our parameter‐by‐date mappings
    param_block = data["properties"]["parameter"]
    # We can use any parameter to grab the list of dates (they all share the same dates)
    dates = list(param_block["T2M"].keys())

    # 3) Build a DataFrame, pulling each parameter’s value by date
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "t2m":            [param_block["T2M"][d]            for d in dates],
        "t2m_max":        [param_block["T2M_MAX"][d]        for d in dates],
        "t2m_min":        [param_block["T2M_MIN"][d]        for d in dates],
        "dewpoint":       [param_block["T2MDEW"][d]         for d in dates],
        "rh2m":           [param_block["RH2M"][d]           for d in dates],
        "ws10m":          [param_block["WS10M"][d]          for d in dates],
        "wd10m":          [param_block["WD10M"][d]          for d in dates],
        "ps":             [param_block["PS"][d]             for d in dates],
        "allsky_sw_dwn":  [param_block["ALLSKY_SFC_SW_DWN"][d] for d in dates],
        "allsky_lw_dwn":  [param_block["ALLSKY_SFC_LW_DWN"][d] for d in dates],
        "prectotcorr":    [param_block["PRECTOTCORR"][d]    for d in dates],
    })

    # 4) Add metadata columns
    df["location"] = location
    df["lat"]      = lat
    df["lon"]      = lon

    # 5) Compute derived features
    #    a) daily temperature range
    df["t2m_range"]   = df["t2m_max"] - df["t2m_min"]

    #    c) day‐of‐year (useful for seasonality features)
    df["day_of_year"] = df["date"].dt.dayofyear

    # 6) Reorder columns to your desired schema
    df = df[[
        "date",
        "location",
        "lat",
        "lon",
        "t2m",
        "t2m_max",
        "t2m_min",
        "t2m_range",
        "dewpoint",
        "rh2m",
        "ws10m",
        "wd10m",
        "ps",
        "allsky_sw_dwn",
        "allsky_lw_dwn",
        "prectotcorr",
        "day_of_year"
    ]]

    return df
