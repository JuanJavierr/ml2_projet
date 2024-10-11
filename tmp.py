# %%

import geopandas as gpd

# %%


df = gpd.read_file("./SHP/mrc_s.shp")

# %%
one_mrc = df[df["MRS_NM_MRC"] == "Rouville"]

# %%
one_mrc.plot()

# %%
