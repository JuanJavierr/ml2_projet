# %%
from utils import *
import numpy as np

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



# %%
def convert_date_to_timeindex(df):
    # Adds a column to df representing the time index
    df["date"] = df["date"].dt.to_period("M")
    min_date = df["date"].min()

    # Convert from days to months
    df['time_index'] = (df['date'].dt.year - min_date.year) * 12 + (df['date'].dt.month - min_date.month)
    return df


df = df[df["date"] < pd.to_datetime("2023-01-01")]
# df = convert_date_to_timeindex(df)
df = df.drop(columns=[ "sector"])
df["log_volume"] = np.log(df.total_kwh)
df = df.sort_values(['sector_mrc', 'date'])

df[["total_kwh", "log_volume", "tavg"]] = df[["total_kwh", "log_volume", "tavg"]].astype(np.float32)


# %%
#### Dataset preparation
import torch
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.05,
    mode='min',
)


series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='log_volume', time_col='date')
temp_series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='tavg', time_col='date')

scaler = Scaler(global_fit=False)
series = scaler.fit_transform(series)

# %%
def train_test_split(series):
    train_series_list = []
    val_series_list = []
    for serie in series:
        train, val = serie.split_before(pd.Timestamp("2022-01-01"))
        train_series_list.append(train)
        val_series_list.append(val)

    return train_series_list, val_series_list

train_series, val_series = train_test_split(series)
train_temp_series, val_temp_series = train_test_split(temp_series)


# %%


from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood


val_len = 12

# model = TCNModel(
#     input_chunk_length=24,
#     output_chunk_length=1,
#     random_state=42,
#     # likelihood=LaplaceLikelihood(),
#     pl_trainer_kwargs={"callbacks": [my_stopper]}
# )

# model.fit(
#         series,
#         verbose=True,
#         epochs=100,
#         past_covariates=[train_temp_series],
#         val_series=val_series,
#         val_past_covariates=[val_temp_series],
#         )


def build_fit_tcn_model(
    in_len,
    out_len,
    kernel_size,
    num_filters,
    weight_norm,
    dilation_base,
    dropout,
    lr,
    include_dayofweek,
    likelihood=None,
    callbacks=None,
):

    # reproducibility
    torch.manual_seed(42)

    # some fixed parameters that will be the same for all models
    BATCH_SIZE = 32
    MAX_N_EPOCHS = 30
    NR_EPOCHS_VAL_PERIOD = 1
    # MAX_SAMPLES_PER_TS = 1000

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
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
    )

    train = [s[: -(2 * val_len)] for s in series]
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = scaler.transform(
        [s[-((2 * val_len) + in_len) : -val_len] for s in series]
    )

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        # max_samples_per_ts=MAX_SAMPLES_PER_TS,
        dataloader_kwargs={"num_workers": num_workers},
    )

    # # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")

    return model


# # %%
# model = build_fit_tcn_model(
#     in_len=24,
#     out_len=1,
#     kernel_size=5,
#     num_filters=5,
#     weight_norm=False,
#     dilation_base=2,
#     dropout=0.2,
#     lr=1e-3,
#     include_dayofweek=True,
# )


# %%
import matplotlib.pyplot as plt

def eval_model(preds, name, train_set=train_series, val_set=val_series):
    smapes = smape(preds, val_set)
    print("{} sMAPE: {:.2f} +- {:.2f}".format(name, np.mean(smapes), np.std(smapes)))

    for i in [10, 50, 100, 150, 250, 350]:
        plt.figure(figsize=(15, 5))
        train_set[i][0 :].plot()
        val_set[i].plot(label="actual")
        preds[i].plot(label="forecast")


# model = model.predict(series=train_series, n=val_len)
# eval_model(model, "tcn")


# %%
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)

def objective(trial):
    callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

    # set input_chunk_length, between 5 and 14 days
    months_in = trial.suggest_int("months_in", 5, 14)

    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    months_out = trial.suggest_int("months_out", 1, months_in - 1)
    out_len = months_out

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    num_filters = trial.suggest_int("num_filters", 5, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])

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
        include_dayofweek=include_dayofweek,
        callbacks=callback,
    )

    # Evaluate how good it is on the validation set
    # Forecast 1-step ahead for each series in the validation set
    preds = model.predict(series=train_series, n=val_len)
    smapes = smape(val_series, preds, n_jobs=-1, verbose=True)
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
