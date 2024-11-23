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
    data = Monthly(location, start=datetime.strptime("2016-01-01", "%Y-%m-%d"), end=datetime.strptime("2023-12-31", "%Y-%m-%d"))

    data = data.fetch()
    data["mrc"] = i

    result.append(data)


# %%

# Get closest MRC for each MRC to fill in missing values
import geopandas as gpd

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat))
gdf = gdf.set_crs("EPSG:4326")
gdf = gdf.to_crs("EPSG:2950")

for i, row in df.iterrows():
    dist_from_mrc = gdf.distance(gdf.loc[i, "geometry"]).sort_values()
    closest_mrc = dist_from_mrc[1:].index[0]
    second_closest_mrc = dist_from_mrc[1:].index[1]

    df.loc[i, "closest_mrc"] = closest_mrc
    df.loc[i, "second_closest_mrc"] = second_closest_mrc
    df.loc[i, "third_closest_mrc"] = dist_from_mrc[1:].index[2]
# %%
# Replace missing values with closest MRC
for i,(mrc, row) in enumerate(df.iterrows()):
    data = result[i]
    closest_mrc = df.loc[mrc, "closest_mrc"]
    second_closest_mrc = df.loc[mrc, "second_closest_mrc"]

    closest_mrc_row_nb = df.index.get_loc(closest_mrc)
    closest_data = result[closest_mrc_row_nb]

    second_closest_mrc_row_nb = df.index.get_loc(second_closest_mrc)
    second_closest_data = result[second_closest_mrc_row_nb]

    third_closest_mrc = df.loc[mrc, "third_closest_mrc"]
    third_closest_mrc_row_nb = df.index.get_loc(third_closest_mrc)
    third_closest_data = result[third_closest_mrc_row_nb]

    for col in data.columns:
        data[col] = data[col].fillna(closest_data[col])
        data[col] = data[col].fillna(second_closest_data[col])
        data[col] = data[col].fillna(third_closest_data[col])


    result[i] = data
# %%

pd.concat(result).to_csv("monthly_avg_temp.csv")
# %%
