import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import numpy as np
import patsy


def load_data():
    df = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")

    return df


def get_consumption_for(df: pd.DataFrame, mrc, sector):
    # Filter
    df = df[(df["MRC_TXT"] == mrc) & (df["SECTEUR"] == sector)]

    # Convert types
    df.loc[:, "Total (kWh)"] = df["Total (kWh)"].astype(float)
    df.loc[:, "ANNEE_MOIS"] = pd.to_datetime(df["ANNEE_MOIS"])
    df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()

    # Standardize column names
    df = df.rename(columns={"Total (kWh)": "total_kwh"})
    df["total_mwh"] = df["total_kwh"] / 1000

    # Set frequency
    df = df.dropna().asfreq("MS")

    # Add helper columns
    df["month"] = df.index.month.astype(str)

    return df.dropna()


def acf_pacf(series: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(series.diff(12).dropna(), lags=40, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(series.diff(12).dropna(), lags=40, ax=axes[1])

    plt.show()


def test_models(df):
    results = []
    y, X = patsy.dmatrices("total_mwh ~ month", data=df, return_type="dataframe")

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


def fit_model(train_df, p, q):

    y, X = patsy.dmatrices("total_mwh ~ month", data=train_df, return_type="dataframe")

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


def forecast(model, df):
    y, X = patsy.dmatrices("total_mwh ~ month", data=df, return_type="dataframe")

    # Get month-ahead forecast for each month in the test set
    forecast = model.get_prediction(start=df.index.min(), end=df.index.max(), exog=X)

    mse = forecast.predicted_mean.sub(y["total_mwh"]).pow(2).mean()
    return forecast, mse


def plot_predictions(test_df, fore):
    import plotly.graph_objects as go

    fore = fore.summary_frame().drop(columns=["mean_se"])

    # Plot forecast (with confidence intervals) against actual data
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(x=test_df.index, y=test_df["total_mwh"], mode="lines", name="Actual")
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


if __name__ == "__main__":
    df = load_data()
    df = get_consumption_for(df, "Maria-Chapdelaine", "RÉSIDENTIEL")  # Abitibi
    train_df = df["2016":"2022"]
    test_df = df["2023":]

    acf_pacf(df["total_mwh"])

    best_p, best_q = test_models(train_df)

    model = fit_model(train_df, best_p, best_q)

    # df["fitted_values"] = model.fittedvalues[12:]
    # plot_residuals(model)

    model.plot_diagnostics()

    fore, mse = forecast(model, test_df)
    fig = plot_predictions(test_df, fore)
    fig.show()
