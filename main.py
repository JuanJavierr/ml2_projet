# %%
import pandas as pd
import plotly.express as px

df = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")

df.head()
# %%
mrc = df[df["MRC_TXT"] == 'Rouville']
mrc.head()

# %%
mrc.plot(x="ANNEE_MOIS", y="Total (kWh)", kind="line", ="SECTEUR")
# %%
def plot_mrc(mrc_name):
    mrc = df[df["MRC_TXT"] == mrc_name]
    plt = px.line(mrc, x="ANNEE_MOIS", y="Total (kWh)", color="SECTEUR")
    plt.show()

# %%

plot_mrc("Rouville")

# %%
