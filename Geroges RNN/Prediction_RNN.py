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
import matplotlib.pyplot as plt
from sklearn import preprocessing
%matplotlib inline

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
# Rouville['Simulation '] = np.random.randint(1, 101, size=96)**

df = Rouville.copy()

# df.drop(columns=['REGION_ADM_QC_TXT','MRC_TXT','ANNEE_MOIS', 'SECTEUR'], axis=1, inplace=True)**
# display(df.head())

# df_open = df[['Total (kWh)']].to_numpy()

# # Normalisation ici....
# df_restant = df.drop(columns=['Total (kWh)'], axis=1)
# scaler = preprocessing.StandardScaler()
# df_restant = scaler.fit_transform(df_restant)

# df = np.concatenate([df_open , df_restant], axis=1)**

df = df[['Total (kWh)']].copy()


df_train =  df[ 0 : int(0.8 * df.shape[0]) ] # df[ 0 : int(0.8 * df.shape[0]) , : ]
df_test =   df[  int(0.8 * df.shape[0]) :  ]   # df[  int(0.8 * df.shape[0]) :  , : ]

print(df_train.shape)
print(df_test.shape)
# %%
# 4. Le generateur
class Generateur(tf.keras.utils.Sequence):

    def __init__(self, dataset, batch_size=8, window_size=11):

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

            X.append( dataset.iloc[ i : i + window_size, : ] )
            y.append( dataset.iloc[i + window_size, 0] )

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
    input_shape = (11, 1),
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
history = model.fit(
    x = Generateur(df_train, batch_size=4),
    epochs = 100,
    validation_data = Generateur(df_test),
)

# Extraire les métriques d'entraînement et de validation
history_dict = history.history
epochs = range(1, len(history_dict['loss']) + 1)

# Graphe des pertes
plt.figure(figsize=(14, 6))

# Pertes (Loss)
plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict['loss'], 'bo-', label='Perte d\'entraînement')
plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Perte de validation')
plt.title('Évolution de la perte')
plt.xlabel('Époques')
plt.ylabel('Perte (Loss)')
plt.legend()
plt.grid(True)

# Mean Absolute Error (MAE)
plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict['mean_absolute_error'], 'bo-', label='MAE d\'entraînement')
plt.plot(epochs, history_dict['val_mean_absolute_error'], 'ro-', label='MAE de validation')
plt.title('Évolution de l\'erreur absolue moyenne (MAE)')
plt.xlabel('Époques')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# Afficher les graphes
plt.tight_layout()
plt.show()
# %%
# Note: On peut rajouter les predicteurs a volonte...


# %% 

# 7. Predictions et visualisations

# 7.1. Générer des prédictions sur l'ensemble des données
full_generator = Generateur(df, batch_size=1)  
full_predictions = model.predict(full_generator)

# 7.2 Récupérer les vraies valeurs de `y` pour tout le dataset
true_values_full = full_generator.y

# %% 
# 8. Visualiser les résultats avec Matplotlib/px

# Créer un DataFrame pour les visualisations
df_results_full = pd.DataFrame({
    "Date": range(len(true_values_full)),
    "Valeurs Réelles": true_values_full.flatten(),
    "Prédictions": full_predictions.flatten()
})

# Utiliser Plotly pour créer un graphique interactif
fig = px.line(
    df_results_full, 
    x="Date", 
    y=["Valeurs Réelles", "Prédictions"], 
    title="Comparaison des Prédictions et des Valeurs Réelles <br> Consommation Energie - Rouville - Agricole",
    labels={"value": "Valeur (kWh)", "variable": "Série"}
)

fig.update_traces(
    selector=dict(name="Prédictions"), 
    line=dict(dash="dot")              
)

fig.show()


# %%
# Fine - tuning Auto keras
import autokeras as ak
import numpy as np

time_steps = 3



# Exemple de fonction pour créer des séquences glissantes
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # Séquence temporelle
        y.append(data[i + time_steps])   # Valeur cible
    return np.array(X), np.array(y)

# Paramètres
time_steps = 10  # Longueur de chaque séquence
vec_len = 1      # Nombre de caractéristiques

# Transformez les données d'entraînement
X_train, y_train = create_sequences(df_train.to_numpy().flatten(), time_steps)

# Reformatez X_train en 3D
X_train = X_train.reshape(X_train.shape[0], time_steps, vec_len)
y_train = y_train.reshape(-1, 1)  # Cible en 2D

print("X_train shape:", X_train.shape)  # (num_samples, time_steps, vec_len)
print("y_train shape:", y_train.shape)  # (num_samples, 1)

# Définir l'entrée avec time_steps et vec_len
input_node = ak.Input(shape=(time_steps, vec_len))

# Ajouter le bloc RNN
output_node = ak.RNNBlock()(input_node)

# Ajouter une tête de régression
output_node = ak.RegressionHead()(output_node)

# Construire le modèle AutoKeras
auto_model = ak.AutoModel(
    inputs=input_node, 
    outputs=output_node, 
    overwrite=True, 
    max_trials=10
)

# Entraîner le modèle
auto_model.fit(X_train, y_train, epochs=100)

# Transformez les données de test
X_test, y_test = create_sequences(df_test.to_numpy().flatten(), time_steps)
X_test = X_test.reshape(X_test.shape[0], time_steps, vec_len)
y_test = y_test.reshape(-1, 1)

# Évaluation du modèle
y_pred = auto_model.predict(X_test)
print("Performance:", auto_model.evaluate(X_test, y_test))

# %%

df_results_full = pd.DataFrame({
    "Date": range(len(y_pred)),
    "Valeurs Réelles": y_test.reshape,
    "Prédictions": y_pred.reshape
})

plt.figure(figsize=(12, 6))

# Courbe des valeurs réelles (en bleu vif)
plt.plot(
    range(len(true_values_full)), 
    true_values_full, 
    label='Valeurs Réelles', 
    color='b',  # Couleur bleu vif
    linewidth=2
)

# Courbe des prédictions (en vert clair, avec un style pointillé)
plt.plot(
    range(len(full_predictions)), 
    full_predictions, 
    label='Prédictions', 
    color='green',  # Couleur vert clair
    linestyle='dotted',
    linewidth=2
)

# Titres, légendes et grille
plt.title('Comparaison des Prédictions et des Valeurs Réelles \n Consommation Energie - Rouville - Agricole', fontsize=14)
plt.xlabel('Temps', fontsize=12)
plt.ylabel('Valeur (kWh)', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.show()


# %%

