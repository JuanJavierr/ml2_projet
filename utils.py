from datetime import datetime
from pathlib import Path

import pandas as pd


def load_data():
    df = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")
    df = df.dropna(subset=["MRC_TXT"])
    df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()
    df = df.rename(columns={"Total (kWh)": "total_kwh", "MRC_TXT": "mrc", "SECTEUR": "sector"})

    df = df.reset_index(drop=True)
    df["ANNEE_MOIS"] = pd.to_datetime(df["ANNEE_MOIS"] + "-01", format="%Y-%m-%d")

    return df


def remove_incomplete_mrc_sectors(df):
    unwanted_mrc_sectors = [
        ("Administration régionale Kativik", "AGRICOLE"),
        ("Administration régionale Kativik", "INDUSTRIEL"),
        ("Caniapiscau", "AGRICOLE"),
        ("Le Golfe-du-Saint-Laurent", "AGRICOLE")
    ]

    for mrc, sector in unwanted_mrc_sectors:
        df = df[~((df["mrc"] == mrc) & (df["sector"] == sector))]


    return df


def interpolate_missing_values(df):
    """Interpolate missing values in the dataset"""
    new_df = pd.DataFrame()

    df["sector_mrc"] = df["sector"] + "_" + df["mrc"]
    for sector_mrc in df["sector_mrc"].unique():
        sector_mrc_df = df[df["sector_mrc"] == sector_mrc].sort_index()
        sector_mrc_df["total_kwh"] = sector_mrc_df["total_kwh"].interpolate(method="index")

        new_df = pd.concat([new_df, sector_mrc_df])

    return new_df


def join_dataframes(df, temp):
    temp["index"] = pd.to_datetime(temp["index"], format="%Y-%m-%d")

    joined_df = df.merge(temp, how="left", left_on=["mrc", "ANNEE_MOIS"], right_on=["mrc", "index"])

    return joined_df


def fetch_temperature_data(mapping):
    from meteostat import Monthly, Point
    
    if Path("data/temperature.csv").exists():
        return pd.read_csv("data/temperature.csv", index_col=0)

    df = mapping.reset_index()
    df = df.rename(columns={"index": "mrc"})

    result = pd.DataFrame()
    for i, row in df.iterrows():
        if row["lat"] is None or row["lng"] is None:
            print(f"Skipping {row['mrc']}")
            continue

        point = Point(row["lat"], row["lng"])
        data = Monthly(point,  start=datetime.strptime("2016-01-01", "%Y-%m-%d"), end=datetime.strptime("2023-12-31", "%Y-%m-%d"))
        data = data.fetch()
        data["mrc"] = row["mrc"]
        if data.empty:
            print(f"No data found for {row['mrc']}")
            # build empty dataframe with same columns and one row per month
            data = pd.DataFrame(columns=["time", "tavg", "mrc"])
            data["time"] = pd.date_range(start="2016-01-01", end="2023-12-31", freq="MS")
            data.index = data["time"]
            data["tavg"] = None
            data["mrc"] = row["mrc"]
        else:
            # Ensure that time goes from 2016-01-01 to 2023-12-31
            data = data.reindex(pd.date_range(start="2016-01-01", end="2023-12-31", freq="MS"))
            data["mrc"] = row["mrc"]


        result = pd.concat([result, data])
        print(f"Done with {row['mrc']}")

    result.to_csv("data/temperature.csv")


    return result

def fill_temperature_data(proximity, temp):

    if Path("data/filled_temp.csv").exists():
        return pd.read_csv("data/filled_temp.csv", index_col=0)

    no_na = temp.dropna(subset=["tavg"])

    filled_result = temp.copy()
    for i, row in temp.iterrows():
        if pd.isna(row["tavg"]):
            closest_mrc = proximity.loc[0, row['mrc']]
            second_closest_mrc = proximity.loc[1, row['mrc']]
            third_closest_mrc = proximity.loc[2, row['mrc']]

            closest_mrc_temp = no_na.loc[(no_na["mrc"] == closest_mrc) & (no_na.index == i), "tavg"].mean()
            second_closest_mrc_temp = no_na.loc[(no_na["mrc"] == second_closest_mrc) & (no_na.index == i), "tavg"].mean()
            third_closest_mrc_temp = no_na.loc[(no_na["mrc"] == third_closest_mrc) & (no_na.index == i), "tavg"].mean()

            global_temp_avg = no_na.loc[no_na.index == i, "tavg"].mean()

            filled_result.loc[i, "tavg"] = closest_mrc_temp
            if pd.isna(closest_mrc_temp):
                filled_result.loc[i, "tavg"] = second_closest_mrc_temp
                if pd.isna(second_closest_mrc_temp):
                    filled_result.loc[i, "tavg"] = third_closest_mrc_temp
                    if pd.isna(third_closest_mrc_temp):
                        print(f"No closest MRC found for {row['mrc']}, index {i}")
                        filled_result.loc[i, "tavg"] = global_temp_avg


    filled_result = filled_result.reset_index()
    filled_result.to_csv("data/filled_temp.csv")

    filled_result = filled_result.drop(columns=["time"])

    return filled_result

def get_proximity_mapping(mapping):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(mapping, geometry=gpd.points_from_xy(mapping.lng, mapping.lat)).set_crs("ESRI:102003")
    result = pd.DataFrame()
    for i, mrc in gdf.iterrows():
        dist_from_mrc = gdf.distance(gdf.loc[i, "geometry"]).sort_values()
        # closest_mrcs = list(dist_from_mrc.index[1:4])

        # index_to_name = dict(zip(gdf.index, gdf.mrc))
        # gdf.loc[i, "closest_mrc"] = index_to_name[closest_mrcs[0]]
        # gdf.loc[i, "second_closest_mrc"] = index_to_name[closest_mrcs[1]]
        # gdf.loc[i, "third_closest_mrc"] = index_to_name[closest_mrcs[2]]
        result[i] = dist_from_mrc.index[1:]

    return result


def fetch_geolocation_data(df):
    """Fetch geolocation (lng, lat) data for MRCs in the dataset"""
    import googlemaps

    if Path("data/geolocation.csv").exists():
        result = pd.read_csv("data/geolocation.csv", index_col=0)
        result.index.name = "mrc"
        return result
    
    gmaps = googlemaps.Client(key='AIzaSyBvKE5dWO_bA27_n2s7DNr0WSw3JfSMOFw')
    mrcs = df["mrc"].unique()
    result = []
    for mrc in mrcs:
        geocode_result = gmaps.geocode(mrc + ", Québec")
        print(geocode_result)
        result.append(geocode_result)

    mapping = {}

    for i, res in enumerate(result):
        found = False
        for comp in res[0]["address_components"]:
            if "administrative_area_level_3" in comp["types"]:
                found = True
                mapping[mrcs[i]] = res[0]["geometry"]["location"]
                break
        if not found:
            print("No match for ", mrcs[i])
            mapping[mrcs[i]] = None


    result = pd.DataFrame(mapping).T
    result.index.name = "mrc"

    result.to_csv("data/geolocation.csv")
            
            
    return result


