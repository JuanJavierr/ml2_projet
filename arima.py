# %%

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import numpy as np
import patsy
import warnings

# from meteostatt import df as weather_df

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Send all print statements to the logger in addition to the console
logger = logging.getLogger()
logger.addHandler(logging.FileHandler("arima.log", mode="w"))


def load_data():
    df = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")
    df = df.dropna(subset=["MRC_TXT"])
    df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()
    df = df.rename(columns={"Total (kWh)": "total_kwh"})

    return df


# %%
def get_consumption_for(df: pd.DataFrame, mrc, sector):
    # # Filter
    df = df[(df["mrc"] == mrc) & (df["sector"] == sector)]
    df = df.set_index(pd.to_datetime(df["date"], format="%Y-%m-%d")).sort_index()
    df["month"] = df.index.month.astype(str)
    return df

    # # Convert types
    # # df.loc[:, "Total (kWh)"] = df["Total (kWh)"].astype(float)
    # df.loc[:, "ANNEE_MOIS"] = pd.to_datetime(df["ANNEE_MOIS"])
    # df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()

    # # Standardize column names
    # df["total_mwh"] = df["total_kwh"] / 1000

    # # Set frequency
    # df = df.dropna().asfreq("MS")

    # # Add helper columns
    # df["month"] = df.index.month.astype(str)

    # weather_df = pd.read_csv("weather.csv")
    # df = pd.merge(df, weather_df, left_index=True, right_index=True)
    return df.dropna()


def acf_pacf(series: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(series.diff(12).dropna(), lags=40, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(series.diff(12).dropna(), lags=40, ax=axes[1])

    plt.show()


def test_models(df, formula):
    results = []
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    for p in range(3):
        for q in range(3):

            if p == 0 and q == 0:
                continue
            model = sm.tsa.statespace.SARIMAX(
                endog=y,
                exog=X,
                order=(p, 1, q),
                seasonal_order=(p, 1, q, 12),
                # enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            results.append([p, q, res.aic])

    best_p, best_q = (
        pd.DataFrame(results, columns=["p", "q", "aic"])
        .set_index(["p", "q"])
        .idxmin()
        .at["aic"]
    )

    return best_p, best_q


def fit_model(train_df, p, q, formula):

    y, X = patsy.dmatrices(formula, data=train_df, return_type="dataframe")

    model = sm.tsa.statespace.SARIMAX(
        endog=y,
        exog=X,
        order=(p, 1, q),
        seasonal_order=(p, 1, q, 12),
        # enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    return res


def plot_residuals(res):
    fig = px.line(
        res.resid[12:],
        title="Residuals",
    )
    fig.show()


def forecast(model, test_df, formula):
    y, X = patsy.dmatrices(formula, data=test_df, return_type="dataframe")
    
    forecast = model.forecast(steps=12, exog=X)
    # Get month-ahead forecast for each month in the test set
    # forecast = model.get_prediction(start="2023-01-01", end="2023-12-01", exog=X)
    forecast.index = y.index
    rmse = (forecast.sub(y["total_kwh"])**2).mean()**0.5
    mape = (forecast.sub(y["total_kwh"]).abs() / y["total_kwh"]).mean()
    return forecast, rmse, mape, y


def plot_predictions(test_df, fore):
    import plotly.graph_objects as go

    fore = fore.summary_frame().drop(columns=["mean_se"])

    # Plot forecast (with confidence intervals) against actual data
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(x=test_df.index, y=test_df["total_kwh"], mode="lines", name="Actual")
    )

    # Add forecasted mean
    fig.add_trace(
        go.Scatter(x=fore.index, y=fore["mean"], mode="lines", name="Forecast")
    )

    # Add confidence intervals as shaded areas
    fig.add_trace(
        go.Scatter(
            x=fore.index.tolist() + fore.index[::-1].tolist(),
            y=fore["mean_ci_upper"].tolist() + fore["mean_ci_lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Confidence Interval",
        )
    )

    fig.update_layout(
        title="Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Total kWh",
    )

    return fig

# %%
if __name__ == "__main__":

    formulas = {
        # "month": "total_kwh ~ month",
        "mean_temp__month": "total_kwh ~ tavg + month",
    }


    # full_df = load_data()
    full_df = pd.read_csv("dataset.csv")
    results_df = pd.DataFrame()
    results_df[["MRC", "SECTOR"]] = full_df[["mrc", "sector"]].drop_duplicates()
    models = {}
    for results_col, formula in formulas.items():
        for mrc in ["Drummond", "Les Etchemins"]:
            for sector in full_df["sector"].unique():
                df = get_consumption_for(full_df, mrc, sector)  # Abitibi
                train_df = df["2016":"2021"]
                validation_df = df["2022":"2022"]
                test_df = df["2023":]

                try:
                    best_p, best_q = test_models(train_df, formula=formula)

                    model = fit_model(train_df, best_p, best_q, formula=formula)

                    # df["fitted_values"] = model.fittedvalues[12:]
                    # plot_residuals(model)

                    # model.plot_diagnostics()

                    fore, rmse, mape, y = forecast(model, validation_df, formula)
                    # fig = plot_predictions(test_df, fore)

                    # with open(f"./plots/forecast_{mrc}.html", "w") as f:
                    #     f.write(fig.to_html())
                    logger.info(f"MRC: {mrc}, SECTOR: {sector},  RMSE: {rmse}, MAPE: {mape}")
                    logger.info(f"Best p: {best_p}, Best q: {best_q}")
                    results_df.loc[((results_df["MRC"] == mrc) & (results_df["SECTOR"] == sector)), results_col+"_rmse"] = rmse
                    results_df.loc[((results_df["MRC"] == mrc) & (results_df["SECTOR"] == sector)), results_col+"_mape"] = mape

                    models[(mrc, sector)] = [model, best_p, best_q]
                    
                
                except Exception as e:
                    logger.error(f"Error for MRC: {mrc}, {sector}, {e}")
                    continue

    results_df.to_csv("results2.csv", index=False)
    
# %%
import plotly.express as px
import plotly.graph_objects as go

for (mrc, sector), (model, best_p, best_q) in models.items():
    df = get_consumption_for(full_df, mrc, sector)  # Abitibi
    train_df = df["2016":"2022"]
    test_df = df["2023":]

    formula = formulas["mean_temp__month"]

    model = fit_model(train_df, best_p, best_q, formula=formula)

    fore, rmse, mape, y = forecast(model, test_df, formula)
    
    # Plot forecast vs actual
    fig = px.line(fore.rename("Prédiction"), title=f"{mrc} - {sector}")
    fig.add_trace(go.Scatter(x=y.index, y=y["total_kwh"], mode="lines", name="Valeurs réelles"))
    fig.update_layout(legend_title_text="Légende", title=f"Consommation mensuelle pour la MRC de {mrc} - Secteur {sector}", yaxis_title="Valeur (kWh)", xaxis_title="Date")
    fig.show()

    y["forecast"] = fore
    y.to_csv(f"arima_predictions/{mrc}_{sector}.csv")


# %%
