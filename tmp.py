# %%
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

    # Set frequency
    df = df.dropna().asfreq("MS")

    # Add helper columns
    df["month"] = df.index.month.astype(str)

    return df.dropna()


df = load_data()
df = get_consumption_for(df, "Maria-Chapdelaine", "RÉSIDENTIEL")  # Abitibi
train_df = df["2016":"2022"]
test_df = df["2023":]


# %%
def acf_pacf(series: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(series.diff(12).dropna(), lags=40, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(series.diff(12).dropna(), lags=40, ax=axes[1])

    plt.show()


acf_pacf(df["total_kwh"])


# %%
def test_models(df):
    aic_full = pd.DataFrame(np.zeros((6, 6), dtype=float))
    mse_full = pd.DataFrame(np.zeros((6, 6), dtype=float))
    y, X = patsy.dmatrices("total_kwh ~ month", data=df, return_type="dataframe")

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
            aic_full.iloc[p, q] = res.aic
            mse_full.iloc[p, q] = (
                res.fittedvalues.sub(y["total_kwh"]).pow(2)[12:].mean()
            )

    print(aic_full)
    print(mse_full)


test_models(train_df)


# %%
def fit_model():

    y, X = patsy.dmatrices("total_kwh ~ month", data=train_df, return_type="dataframe")

    model = sm.tsa.statespace.SARIMAX(
        endog=y,
        exog=X,
        order=(1, 1, 2),
        seasonal_order=(1, 1, 2, 12),
        # enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    return res


model = fit_model()


def plot_residuals(res):
    fig = px.line(
        res.resid[12:],
        title="Residuals",
    )
    fig.show()


def plot_predictions(df):
    fig = px.line(
        df,
        x=df.index,
        y=["total_kwh", "fitted_values"],
        title="Total consumption vs forecast",
    )
    fig.show()


df["fitted_values"] = model.fittedvalues[12:]
plot_predictions(df)
# plot_residuals(model)

# %%

model.plot_diagnostics()


# %%
def forecast(model, df):
    y, X = patsy.dmatrices("total_kwh ~ month", data=df, return_type="dataframe")

    # Get month-ahead forecast for each month in the test set
    forecast = model.get_prediction(start=df.index.min(), end=df.index.max(), exog=X)
    return forecast


fore = forecast(model, test_df)
fore = fore.summary_frame().drop(columns=["mean_se"])
# Plot forecast (with confidence intervals) against actual data
fig = px.line(
    data_frame=fore.join(test_df),
    x=fore.index,
    y=["total_kwh", "mean", "mean_ci_lower", "mean_ci_upper"],
    title="Forecast",
)
fig.show()


# %%
def plot_predictions():
    import plotly.graph_objects as go

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

    fig.show()


plot_predictions()
# %%
