#%% 
# !pip install openpyxl
# !pip install px
# !pip install gpd
# !pip install streamlit
# !pip install pydeck
# !pip install geopandas

#%% 
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# %% 
# 1. Lire le fichier csv
data = pd.read_excel("consommation-historique-mrc-11mars2024.xlsx")

# %%
# 2. Afficher le contenu pour l'analyser
display(data.head(25))
print(f"Les MRC presents : {data['MRC_TXT'].unique()}")

mask1 = data['MRC_TXT'] == 'Rouville' # focus sur Rouville
mask2 = data['SECTEUR'] == 'AGRICOLE' # focus sur Besoin Agricole
Rouville = data[mask1 & mask2]
Rouville.shape

display(Rouville.shape)
display(Rouville.head())

#%%
# 3. Retire les colonnes qui presentent des donnees non-necessaires ou aberrantes (Features)
Rouville['Simulation '] = np.random.randint(1, 101, size=96)

df = Rouville.copy()

df.drop(columns=['REGION_ADM_QC_TXT','MRC_TXT','ANNEE_MOIS', 'SECTEUR'], axis=1, inplace=True)
display(df.head())

df_open = df[['Total (kWh)']].to_numpy()

# Normalisation ici....
df_restant = df.drop(columns=['Total (kWh)'], axis=1)
scaler = preprocessing.StandardScaler()
df_restant = scaler.fit_transform(df_restant)

df = np.concatenate([df_open , df_restant], axis=1)


df_train = df[ 0 : int(0.8 * df.shape[0]) , : ]
df_test = df[  int(0.8 * df.shape[0]) :  , : ]

print(df_train.shape)
print(df_test.shape)
# %%
# 4. Le generateur
class Generateur(tf.keras.utils.Sequence):

    def __init__(self, dataset, batch_size=8, window_size=7):

        # Normalise le DATASET

        self.X , self.y = self.slide_window(dataset, window_size)

        self.batch_size = batch_size


    def __len__(self):
        
        return self.X.shape[0] // self.batch_size

    def __getitem__(self, idx):

        batch_x = self.X[ self.batch_size * idx : (idx+1) * self.batch_size , : ]
        batch_y = self.y[ idx * self.batch_size : (idx + 1) * self.batch_size]

        return np.asarray(batch_x).astype(np.float32) , np.asarray(batch_y).astype(np.float32)

    def slide_window(self, dataset, window_size=7):

        X, y = [], []

        # Trouver toutes les sequences de 'window_size' donnees, dans le dataset,
        # ainsi que leur 'y' (target)
        for i in range( dataset.shape[0] - window_size ):

            X.append( dataset[ i : i + window_size, : ] )
            y.append( dataset[i + window_size, 0] )

        return np.asarray(X).astype(np.float32) , np.asarray(y).astype(np.float32)

# %%
# 5. Creation du modele LSTM
def create_model(input_shape, output_shape):

    # 1. Definir une couche d'entrée (Input)
    input = tf.keras.Input(shape=input_shape)

    output = tf.keras.layers.Dense(128, activation="relu")(input) # Hidden Layer 1
    output = tf.keras.layers.Dense(256, activation="relu")(output) # Hidden Layer 2

    # Cellule(s) memoire(s)
    output = tf.keras.layers.LSTM(512)(output)

    # Descendre l'entonnoir, en diminuant le nombre de neurones par couche cachée
    output = tf.keras.layers.Dense(256, activation="relu")(output) # Hidden Layer 3
    output = tf.keras.layers.Dense(128, activation="relu")(output) # Hidden Layer 4

    # Creer notre cellule de sortie
    output = tf.keras.layers.Dense(output_shape, activation="relu")(output)

    # Retourner le modele qu'on vient de créer
    return tf.keras.Model(inputs = input , outputs = output)


# Creer une instance du modele
model = create_model(
    input_shape = (7, 2),
    output_shape = 1)

model.summary()

# %%
# 6. Entrainnement 
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = "mean_squared_error",
    metrics = [ "mean_absolute_error" ]
)


# Iterer pour faire l'entrainement
model.fit(
    x = Generateur(df_train, batch_size=4),
    epochs = 100,
    validation_data = Generateur(df_test),
)
# %%
# Note: On peut rajouter les predicteurs a volonte...