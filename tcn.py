# %%
import numpy as np
import optuna
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, rmse
from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (plot_contour, plot_optimization_history,
                                  plot_param_importances)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import *

# %%

def get_data(omit_last_year=False):
    #### Data loading and preprocessing
    df = load_data()
    mapping = fetch_geolocation_data(df)

    mapping.loc["Les Appalaches", "lat"] = 46.374163
    mapping.loc["Les Appalaches", "lng"] = -70.440328

    temp = fetch_temperature_data(mapping)
    proximity = get_proximity_mapping(mapping)
    temp = fill_temperature_data(proximity ,temp)

    df = join_dataframes(df, temp)
    df = remove_incomplete_mrc_sectors(df)
    df = interpolate_missing_values(df)

    df = df.drop(columns=["REGION_ADM_QC_TXT", "index", "tmin", "tmax", "prcp", "wspd", "pres", "tsun", "time", "mrc"])
    df = df.rename(columns={"ANNEE_MOIS": "date"})


    df = df.drop(columns=[ "sector"])
    df["log_volume"] = np.log(df.total_kwh)
    df = df.sort_values(['sector_mrc', 'date'])

    df[["total_kwh", "log_volume", "tavg"]] = df[["total_kwh", "log_volume", "tavg"]].astype(np.float32)

    if omit_last_year:
        df = df[df["date"] < pd.to_datetime("2023-01-01")]

    return df


# %%
df = get_data(omit_last_year=False) # Don't omit last year since we're testing

# %%
#### Dataset preparation
series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='log_volume', time_col='date')
temp_series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='tavg', time_col='date')

# scaler = Scaler()
# temp_scaler = Scaler()
# series = scaler.fit_transform(series)
# temp_series = temp_scaler.fit_transform(temp_series)

# %%

def build_fit_tcn_model(
    in_len,
    out_len,
    kernel_size,
    num_filters,
    weight_norm,
    dilation_base,
    dropout,
    lr,
    num_layers,
    likelihood=None,
    callbacks=None,
):

    # reproducibility
    torch.manual_seed(42)

    # some fixed parameters that will be the same for all models
    BATCH_SIZE = 32
    MAX_N_EPOCHS = 10
    NR_EPOCHS_VAL_PERIOD = 1
    # MAX_SAMPLES_PER_TS = 1000

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=3)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    # # optionally also add the day of the week (cyclically encoded) as a past covariate
    # encoders = {"cyclic": {"past": ["dayofweek"]}} if include_dayofweek else None

    # build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        kernel_size=kernel_size,
        num_filters=num_filters,
        num_layers=num_layers,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        # add_encoders=encoders,
        likelihood=likelihood,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
        add_encoders={"transformer": Scaler()}
    )

    train = [s[: -12] for s in series]
    temp_train = [s[: -12] for s in temp_series]
    print(f"Training max date is: {train[0].end_time()}")
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = [s[-18 : ] for s in series]
    model_val_temp = [s[-18 : ] for s in temp_series]
    print("Validation max date is: ", model_val_set[0].end_time())

    # train the model
    model.fit(
        series=train,
        past_covariates=temp_train,
        val_series=model_val_set,
        val_past_covariates=model_val_temp,
        # max_samples_per_ts=MAX_SAMPLES_PER_TS,
        dataloader_kwargs={"num_workers": num_workers},
    )

    # # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")

    return model


# %%
model = build_fit_tcn_model(
    in_len=12,
    out_len=1,
    kernel_size=8,
    num_filters=24,
    weight_norm=True,
    dilation_base=4,
    dropout=0.4,
    lr=2e-4,
    num_layers=3,
)


# %%
import matplotlib.pyplot as plt

def evaluate(model, series, temp_series):
    # series = [s for s in series if s.static_covariates["sector_mrc"].str.startswith(sector).all()]
    # temp_series = [s for s in temp_series if s.static_covariates["sector_mrc"].str.startswith(sector).all()]
    # print(f"Evaluating sector {sector} with {len(series)} series")
    preds = model.historical_forecasts(
        series=series,
        past_covariates=temp_series,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        start=pd.Timestamp("2023-01-01"),
        verbose=False
    )

    # # Reverse scaling
    # series = scaler.inverse_transform(series)
    # preds = scaler.inverse_transform(preds)

    # Reverse log transformation
    series = [s.map(np.exp) for s in series]
    preds = [s.map(np.exp) for s in preds]

    smapes = mape(series, preds)
    rmses = rmse(series, preds)
    print("MAPE: {:.2f} +- {:.2f}".format(np.mean(smapes), np.std(smapes)))
    print("RMSE: {:.2f} +- {:.2f}".format(np.mean(rmses), np.std(rmses)))

    for i in np.random.choice(range(len(series)), 20):
        plt.figure(figsize=(10, 6))
        series[i].plot(label="actual")
        preds[i].plot(label="forecast")
        plt.title(f"MAPE: {smapes[i]:.2f} - Sector: {series[i].static_covariates['sector_mrc'].iloc[0]}")
        plt.legend()
        # Show dots  and lines
        plt.scatter(series[i].time_index, series[i].values(), color='black', s=10)
        # Disable scientific notation
        plt.ticklabel_format(style='plain', axis='y')
        plt.ylim(0, 1.1 * max(series[i].values()))
        plt.show()

    return smapes, rmses


mapes, rmses = evaluate(model, series, temp_series)
sectors = [s.static_covariates["sector_mrc"].iloc[0].split("_")[0] for s in series]
# %%
# disable pandas scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

results = pd.DataFrame(dict(sector=sectors, mape=mapes, rmse=rmses))

print(results.groupby("sector").mean())
print(f"Globally, {results[["mape", "rmse"]].mean()}")



# %%


def get_forecast(mrc, series, temp_series, model):
    series = [s for s in series if s.static_covariates["sector_mrc"].str.endswith(mrc).all()]
    temp_series = [s for s in temp_series if s.static_covariates["sector_mrc"].str.endswith(mrc).all()]

    preds = model.historical_forecasts(
        series=series,
        past_covariates=temp_series,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        start=pd.Timestamp("2023-01-01"),
        verbose=False
    )

    # # Reverse scaling
    # series = scaler.inverse_transform(series)
    # preds = scaler.inverse_transform(preds)

    # Reverse log transformation
    series = [s.map(np.exp) for s in series]
    preds = [s.map(np.exp) for s in preds]

    

    return preds


drummond_preds = get_forecast("Drummond", series, temp_series, model)
lesetchemins_preds = get_forecast("Les Etchemins", series, temp_series, model)

# %%
results_df = pd.DataFrame()
for i in drummond_preds:
    results_df[i.static_covariates_values()[0][0]] = i.pd_dataframe()

for i in lesetchemins_preds:
    results_df[i.static_covariates_values()[0][0]] = i.pd_dataframe()

results_df.to_csv("tcn_preds.csv")
# %%
#### HHyperparameter optimization

df = get_data(omit_last_year=True) # Omit last year since we're optimizing

#### Dataset preparation
series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='log_volume', time_col='date')
temp_series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='tavg', time_col='date')

scaler = Scaler(global_fit=False)
temp_scaler = Scaler()
series = scaler.fit_transform(series)
temp_series = temp_scaler.fit_transform(temp_series)



def objective(trial):
    # callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    callback = []
    months_in = 12 # trial.suggest_int("months_in", 3, 12)

    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    months_out = 1 #trial.suggest_int("months_out", 1, 3)
    out_len = months_out

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 3, 11)
    num_filters = trial.suggest_int("num_filters", 6, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 0.0001, 0.005, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    # include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])

    # Force kernel_size to be smaller than in_len
    kernel_size = min(kernel_size, months_in) - 1

    # build and train the TCN model with these hyper-parameters:
    model = build_fit_tcn_model(
        in_len=months_in,
        out_len=out_len,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        lr=lr,
        num_layers=num_layers,
        callbacks=callback,
    )

    # Evaluate how good it is on the validation set
    # Forecast 1-step ahead for each series in the validation set


    # full_series = []
    # for i, train_serie in enumerate(train_series):
    #     full_series.append(train_serie.append(val_series[i]))
    
    preds = model.historical_forecasts(
        series=series,
        past_covariates=temp_series,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        start=pd.Timestamp("2022-01-01"),
        verbose=True,
    )
    smapes = smape(series, preds)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")

# %%

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


study = optuna.create_study(direction="minimize")

study.optimize(objective, timeout=7200, callbacks=[print_callback])

# Finally, print the best value and best hyperparameters:
print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

# %%
