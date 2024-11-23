# %%
import googlemaps
from arima import load_data

gmaps = googlemaps.Client(key='AIzaSyBvKE5dWO_bA27_n2s7DNr0WSw3JfSMOFw')

df = load_data()

mrcs = df["MRC_TXT"].unique()
# %%
result = []
for mrc in mrcs:
    geocode_result = gmaps.geocode(mrc + ", Québec")
    print(geocode_result)
    result.append(geocode_result)
# %%
mapping = {}
for i, res in enumerate(result):
    found = False
    for comp in res[0]["address_components"]:
        if "administrative_area_level_3" in comp["types"]:
            # print(comp["long_name"])
            # print(mrcs[i])
            found = True
            mapping[mrcs[i]] = res[0]["geometry"]["location"]
            break
    if not found:
        print("No match for ", mrcs[i])
        mapping[mrcs[i]] = None
# %%
import pandas as pd
pd.DataFrame(mapping).T.to_csv("geolocation.csv")
# %%
