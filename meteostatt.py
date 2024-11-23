# %%
import pandas as pd
from meteostat import Monthly, Point
from datetime import datetime

df = pd.read_csv("geolocation.csv", index_col=0)
df = df.reset_index()
df = df.rename(columns={"index": "mrc"})

# %%
result = pd.DataFrame()
for i,  row in df.iterrows():

    location = Point(row["lat"], row["lng"])

    # Fetch average monthly data
    data = Monthly(location, start=datetime.strptime("2016-01-01", "%Y-%m-%d"), end=datetime.strptime("2023-12-31", "%Y-%m-%d"))

    data = data.fetch()
    # print(i, index)
    data["mrc"] = row["mrc"]
    data["lat"] = row["lat"]
    data["lng"] = row["lng"]

    if data.empty:
        print(f"Empty data for {row['mrc']}")
        data = pd.DataFrame({"mrc": row["mrc"], "lat": row["lat"], "lng": row["lng"], "tavg": [None]}, index=pd.date_range(start="2016-01-01", end="2023-12-31", freq="MS"))

    result = pd.concat([result, data])

result["is_missing"] = result["tavg"].isna()

# %%

# Get closest MRC for each MRC to fill in missing values
import geopandas as gpd

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat))

# Remove MRCs for which we don't have temp data
# gdf = gdf[~gdf.index.isin(["Témiscouata", "Caniapiscau", "Manicouagan", "Avignon", "Antoine-Labelle"])]

gdf = gdf.set_crs("EPSG:4326")
gdf = gdf.to_crs("EPSG:2950")

# %%
for i, mrc in gdf.iterrows():
    dist_from_mrc = gdf.distance(gdf.loc[i, "geometry"]).sort_values()
    indexes_of_closest_mrc = list(dist_from_mrc.index[1:4])

    index_to_name = dict(zip(gdf.index, gdf.mrc))
    gdf.loc[i, "closest_mrc"] = index_to_name[indexes_of_closest_mrc[0]]
    gdf.loc[i, "second_closest_mrc"] = index_to_name[indexes_of_closest_mrc[1]]
    gdf.loc[i, "third_closest_mrc"] = index_to_name[indexes_of_closest_mrc[2]]

# %%
# Replace missing values with closest MRC
print(result.isna().sum())
filled_result = result.copy()
for i, row in result.iterrows():
    if pd.isna(row["tavg"]):
        closest_mrc = gdf[gdf['mrc'] == row["mrc"]]["closest_mrc"].iloc[0]
        second_closest_mrc = gdf[gdf['mrc'] == row["mrc"]]["second_closest_mrc"].iloc[0]
        third_closest_mrc = gdf[gdf['mrc'] == row["mrc"]]["third_closest_mrc"].iloc[0]

        closest_mrc_temp = None
        if closest_mrc is not None and pd.isna(closest_mrc_temp):
            closest_mrc_temp = result.loc[(result["mrc"] == closest_mrc) & (result.index == i), "tavg"].mean()
        if second_closest_mrc is not None and pd.isna(closest_mrc_temp):
            closest_mrc_temp = result.loc[(result["mrc"] == second_closest_mrc) & (result.index == i), "tavg"].mean()
        if third_closest_mrc is not None and pd.isna(closest_mrc_temp):
            closest_mrc_temp = result.loc[(result["mrc"] == third_closest_mrc) & (result.index == i), "tavg"].mean()
        if pd.isna(closest_mrc_temp):
            closest_mrc_temp = result.loc[(result["mrc"] == "Montcalm") & (result.index == i), "tavg"].mean()
        if pd.isna(closest_mrc_temp):
            print(f"No closest MRC found for {row['mrc']}, index {i}")
            raise ValueError("No closest MRC found")
        filled_result.loc[i, "tavg"] = closest_mrc_temp

print(filled_result.isna().sum())

# %%


def fill_missing_values(temp):
    from arima import load_data
    import pandas as pd

    df = load_data()
    df = df.set_index(["MRC_TXT", "SECTEUR", pd.to_datetime(df.index, format="%Y-%m")])


    temp = temp.set_index(["mrc", pd.to_datetime(temp.index)])
    temp = temp[["tavg"]]


    # change df index names
    df.index.names = ["mrc", "sector", "date"]
    temp.index.names = ["mrc", "date"]

    joined_df = df.join(temp, how="left", on=["mrc", "date"])
    joined_df = joined_df.drop(columns=["REGION_ADM_QC_TXT", "ANNEE_MOIS"])
    # get smallest date in temp
    avg_temp_per_month = temp.groupby("date").mean()
    joined_df = joined_df.join(avg_temp_per_month, how="left", on="date", rsuffix="_avg")
    joined_df["tavg"] = joined_df["tavg"].fillna(joined_df["tavg_avg"])
    joined_df = joined_df.drop(columns=["tavg_avg"])

    return joined_df

final_df = fill_missing_values(filled_result)
final_df.to_csv("dataset.csv")

# %%
