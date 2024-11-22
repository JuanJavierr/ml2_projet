import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import pydeck as pdk
import arima

df = arima.load_data()
geo_df = gpd.read_file("./SHP/mrc_s.shp")

# Reduce polygon's complexity
geo_df["geometry"] = geo_df.simplify(0.001)

mrc = geo_df[geo_df["MRS_NM_MRC"] == "Rouville"]


def plot_mrc(mrc_name):
    mrc = df[df["MRC_TXT"] == mrc_name]
    plt = px.line(
        mrc,
        x="ANNEE_MOIS",
        y="Total (kWh)",
        color="SECTEUR",
        title="Consommation d'électricité par secteur pour la MRC: " + str(mrc_name),
    )
    return plt


def map_mrc(mrc_name):
    mrc = geo_df
    center = mrc.geometry.centroid
    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                id="geojson",
                data=mrc,
                auto_highlight=True,
                opacity=0.5,
                get_fill_color=[0, 0, 255],
                stroked=True,
                pickable=True,
            ),
        ],
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=float(center.y.iloc[0]),
            longitude=float(center.x.iloc[0]),
            zoom=4,
            pitch=0,
        ),
        tooltip={
            "html": "<b>{MRS_NM_MRC}</b>",
            "style": {"color": "white"},
        },
    )
    return deck


st.set_page_config(layout="wide")
st.title("Consommation énergétique par secteur pour les MRC du Québec")

mrc_list = df["MRC_TXT"].unique()
mrc_name = st.selectbox("Choose a MRC", mrc_list, index=2)


col1, col2 = st.columns(2)
with col2:
    event = st.pydeck_chart(
        map_mrc(mrc_name),
        on_select="rerun",
    )
    event.selection

with col1:
    st.plotly_chart(plot_mrc(mrc_name))


sector_df = arima.get_consumption_for(df, mrc_name, "RÉSIDENTIEL")
train_df = sector_df["2016":"2022"]
test_df = sector_df["2023":]
model = arima.fit_model(train_df, 1, 2)
fore, mse = arima.forecast(model, test_df)
fig = arima.plot_predictions(test_df, fore)
st.plotly_chart(fig)
st.write("Mean Squared Error: ", mse)
