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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go 
import random

from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

%matplotlib inline

# %% 1. Lire le fichier csv
data = pd.read_csv('dataset.csv')
data = data[(data['mrc'] == 'Drummond') | (data['mrc'] == 'Les Etchemins')] # on selectionne les mrc souhaitees
data['mrc'].unique()

# %%
# 2. Afficher le contenu pour l'analyser
data_dropped = data.drop(['mrc','sector','date'], axis=1)
display(data.head())

# Afficher les MRC disponibles
mrc_list = data['mrc'].unique()
print(f"MRC disponibles : {mrc_list}")

#%%
# 3. Retire les colonnes qui presentent des donnees non-necessaires ou aberrantes (Features)

def preprocess_data(data, mrc_sector='AGRICOLEAbitibi'):
    """Prépare les données pour un MRC donné."""
    mask = data['sector_mrc'] == mrc_sector
    filtered_data = data[mask].copy()

    # Supprimer les colonnes inutiles
    filtered_data = filtered_data[['total_kwh', 'tavg']].copy()

    # Normalisation et transformations
    scaler = preprocessing.StandardScaler()
    filtered_data['tavg'] = scaler.fit_transform(filtered_data[['tavg']])
    filtered_data['tavg_diff'] = filtered_data['tavg'].diff().fillna(0)
    filtered_data['total_kwh'] = np.log(filtered_data['total_kwh'])
    filtered_data['total_kwh_diff'] = filtered_data['total_kwh'].diff().fillna(0)
    
    filtered_data_footer = filtered_data.iloc[int(0.875 * filtered_data.shape[0]):, :]
    filtered_data_header = filtered_data.iloc[ : 84]
    
    # Split train/test
    df_train = filtered_data.iloc[0:int(0.75 * filtered_data.shape[0]) ]
    df_test = filtered_data.iloc[int(0.75 * filtered_data.shape[0]) :  ] 

    df_predictions = filtered_data_footer.copy()

    print("Entrainement")
    display(df_train.head())
    print("Validation")
    display(df_test.head())
    print("Prédictions")
    display(df_predictions.head())
    
    return df_train, df_test, df_predictions


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
    input_shape = (11, 4),
    output_shape = 1)

model.summary()

# %%
# 6. Entrainnement 

# Fonction pour calculer le MAPE
def calculate_mape(true_values, predicted_values):
    result = np.mean(np.abs((true_values - predicted_values) / true_values))
    return tf.convert_to_tensor(result)

class MAPE(tf.keras.metrics.Metric):
    def __init__(self, name="mape", **kwargs):
        super(MAPE, self).__init__(name=name, **kwargs)
        # Initialisation de l'état de la métrique
        self.mape = self.add_weight(name="mape", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calcul du MAPE (Mean Absolute Percentage Error)
        mape_value = np.mean(tf.abs((y_true - y_pred) / y_true))
        # Mettre à jour la métrique
        self.mape.assign_add(mape_value)

    def result(self):
        # Retourner la valeur actuelle de la métrique
        return self.mape

    def reset_state(self):
        # Réinitialiser l'état de la métrique (utile entre chaque époque)
        self.mape.assign(0.)

mrc_sector_list = data['sector_mrc'].unique()

early_stopping = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=10,          
    restore_best_weights=True  
)


# Préparer une liste pour stocker les résultats (MAPE, valeurs réelles, prédictions)
results = []
all_predictions = {}
predictions_unseen_data = {}


all_predictions = {
    "Date": list(range(len(data['date'].unique()))),  # Date commune à toutes les MRC
    "MRCs": {}  # Un sous-dictionnaire pour chaque MRC
}

# Boucle sur chaque combinaison de MRC et secteur
for mrc in mrc_sector_list:
    try:
        print(f"Entraînement pour MRC_Secteur: {mrc}")
        df_train, df_test, df_predictions = preprocess_data(data, mrc_sector=mrc)       
        # Créer des générateurs
        train_generator = Generateur(df_train, batch_size=4)
        test_generator = Generateur(df_test, batch_size=4)
                
        # Réinitialiser le modèle avant chaque entraînement
        model = create_model(input_shape=(11, df_train.shape[1]), output_shape=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mean_squared_error",
            metrics=["mean_absolute_error", MAPE()], 
            run_eagerly=True
        )
        
        # Entraîner le modèle
        history = model.fit(
            x=train_generator,
            epochs=50,
            validation_data=test_generator,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Sauvegarder les métriques ou le modèle
        # model.save(f"model_{mrc}.h5")
        # Extraire les métriques d'entraînement et de validation
        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)

        # Graphe des pertes
        plt.figure(figsize=(14, 6))

        # Pertes (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict['loss'], 'bo-', label='Perte d\'entraînement')
        plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Perte de validation')
        plt.title(f'Évolution de la Loss - {mrc}')
        plt.xlabel('Époques')
        plt.ylabel('Perte (Loss)')
        plt.legend()
        plt.grid(True)

        # Mean Absolute Error (MAE)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict['mape'], 'bo-', label='MAE d\'entraînement')
        plt.plot(epochs, history_dict['val_mape'], 'ro-', label='MAE de validation')
        plt.title(f'Évolution de l\'erreur absolue moyenne (MAE) - {mrc}')
        plt.xlabel('Époques')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        # Afficher les graphes
        plt.tight_layout()
        plt.savefig(f"Resultats_Training/training_metrics_{mrc}.png", format='png')
        plt.show()
        
        data_jumelage = pd.concat([df_train, df_test,df_predictions],axis=0)

        # Prédictions sur l'ensemble des données
        all_inputs = Generateur(data_jumelage, batch_size=1)
        predictions_all_inputs = model.predict(all_inputs)
        predictions_all_inputs = np.exp(predictions_all_inputs)
        
        true_values_all_full = all_inputs.y.flatten()
        true_values_all_full = np.exp(true_values_all_full)
        
        # Calcul du MAE
        mae = mean_absolute_error(true_values_all_full, predictions_all_inputs)

        # Calcul du RMSE
        rmse = np.sqrt(mean_squared_error(true_values_all_full, predictions_all_inputs))
        
        # Mape
        mape = calculate_mape(true_values_all_full, predictions_all_inputs)
               
        # Créer un DataFrame pour les visualisations
        
        df_results_full = pd.DataFrame({
            "Date": range(len(true_values_all_full)),
            "Valeurs Réelles": true_values_all_full.flatten(),
            "Prédictions": predictions_all_inputs.flatten()
        })
        
        # Ajouter les données pour une MRC spécifique
        all_predictions["MRCs"][mrc] = {
            "Valeurs Réelles": true_values_all_full.flatten().tolist(),
            "Prédictions": predictions_all_inputs.flatten().tolist(),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape.numpy()
        }
        
        # Préparer les données pour le tableau
        error_summary = {
            "MRC": [],
            "MAE": [],
            "RMSE": [],
            "MAPE": []
        }

        # Parcourir le dictionnaire pour extraire les erreurs
        for mrc, metrics in all_predictions["MRCs"].items():
            error_summary["MRC"].append(mrc)
            error_summary["MAE"].append(metrics["MAE"])
            error_summary["RMSE"].append(metrics["RMSE"])
            error_summary["MAPE"].append(metrics["MAPE"])

        # Définir les plages de couleurs
        colors = ["black", "blue", "green"]

        # Diviser les données en segments selon les plages
        segments = [
            {"start": 0, "end": 71, "color": colors[0]},
            {"start": 72, "end": 83, "color": colors[1]},
            {"start": 84, "end": len(df_results_full) - 1, "color": colors[2]},
        ]

        # Créer le graphique
        fig = go.Figure()

        # Ajouter les segments pour les Valeurs Réelles
        for segment in segments:
            start, end, color = segment.values()
            fig.add_trace(
                go.Scatter(
                    x=df_results_full["Date"][start:end + 1],
                    y=df_results_full["Valeurs Réelles"][start:end + 1],
                    mode="lines",
                    name="Valeurs Réelles",
                    line=dict(color=color)
                )
            )

        # Ajouter les segments pour les Prédictions
        for segment in segments:
            start, end, color = segment.values()
            fig.add_trace(
                go.Scatter(
                    x=df_results_full["Date"][start:end + 1],
                    y=df_results_full["Prédictions"][start:end + 1],
                    mode="lines",
                    name="Prédictions",
                    line=dict(color=color, dash="dot")
                )
            )

        # Mettre à jour les titres et étiquettes
        fig.update_layout(
            title="Comparaison des Prédictions et des Valeurs Réelles",
            xaxis_title="Date",
            yaxis_title="Valeur (kWh)",
            legend_title="Séries"
        )

        # # Sauvegarder le graphique sous forme d'image
        # fig.write_image("Resultats_Training/comparaison_predictions_valeurs_reelles.png", format="png")

        # Afficher le graphique
        fig.show()
        
        
        



    except ValueError as e:
        print(f"Erreur pour {mrc}: {e}")



#%%
# Créer un DataFrame Pandas
error_df = pd.DataFrame(error_summary)

# Afficher le tableau
print(error_df)

# Sauvegarder le tableau en CSV si nécessaire
error_df.to_csv("Resultats_Training/error_summary.csv", index=False)



#%%

#%%