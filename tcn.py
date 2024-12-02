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

df = convert_date_to_timeindex(df)
df = df.drop(columns=["date", "sector"])
df["log_volume"] = np.log(df.total_kwh)
# %%
#### Dataset preparation
from pytorch_forecasting import TimeSeriesDataSet
import lightning.pytorch as pl
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger



BATCH_SIZE = 32

max_prediction_length = 6
max_encoder_length = 24
training_cutoff = df["time_index"].max() - max_prediction_length




training = TimeSeriesDataSet(
    data = df[lambda x: x.time_index <= training_cutoff],
    time_idx = "time_index",
    group_ids = ["sector_mrc"],
    target = "total_kwh",
    allow_missing_timesteps=False,
    # add_target_scales=True,
    max_encoder_length=max_encoder_length,
    min_encoder_length=max_encoder_length // 2,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    target_normalizer=GroupNormalizer(
        groups=["sector_mrc"],
        transformation="softplus"
    )

)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)
