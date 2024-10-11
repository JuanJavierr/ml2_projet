import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")


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


mrc_list = df["MRC_TXT"].unique()
mrc_name = st.selectbox("Choose a MRC", mrc_list)

if mrc_name is not None:
    st.plotly_chart(plot_mrc(mrc_name))
