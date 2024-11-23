# %%
import pandas as pd
from meteostat import Monthly, Point
from datetime import datetime

df = pd.read_csv("geolocation.csv", index_col=0)
# %%
result = []
for i, row in df.iterrows():

    location = Point(row["lat"], row["lng"])

    # Fetch average monthly data
    data = Monthly(location, start=datetime.strptime("2018-01-01", "%Y-%m-%d"), end=datetime.strptime("2023-12-31", "%Y-%m-%d"))

    data = data.fetch()
    data["mrc"] = i

    result.append(data)

# # Extract average monthly temperatures in Celsius
# if not data.empty:
#     monthly_avg_temp = data['tavg'].ffill()

    
#     # Group by month to calculate averagesmon
#     df = monthly_avg_temp.resample("MS").mean()
#     df = df.rename("mean_temp")
    

#     # Display the results
#     # print(df)
# else:
#     print("No data available for the specified location and period.")


# %%
pd.concat(result).to_csv("monthly_avg_temp.csv")
# %%
