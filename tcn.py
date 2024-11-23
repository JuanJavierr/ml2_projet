# %%
from arima import load_data
import pandas as pd

df = load_data()
df = df.set_index(["MRC_TXT", "SECTEUR", df.index])


# %%
temp = pd.read_csv("monthly_avg_temp.csv", index_col=0)
temp = temp.set_index(["mrc", temp.index])
temp = temp[["tavg"]]

# interpolate missing values
temp = temp.groupby("mrc").apply(lambda x: x.interpolate(method="linear")).droplevel
# %%

df = df.join(temp)
# %%
# change df index names
df.index.names = ["mrc", "sector", "date"]
temp.index.names = ["mrc", "date"]

# %%
df.join(temp)
# %%
