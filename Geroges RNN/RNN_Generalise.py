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

# %% 1. Lire le fichier csv
data = pd.read_csv('dataset.csv')

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
    
    # Supprimer les colonnes inutiles (adapter selon vos besoins)
    filtered_data = filtered_data[['total_kwh', 'tavg']].copy()
    
    # Normalisation et transformations
    scaler = preprocessing.StandardScaler()
    filtered_data['tavg'] = scaler.fit_transform(filtered_data[['tavg']])
    filtered_data['tavg_diff'] = filtered_data['tavg'].diff().fillna(0)
    filtered_data['total_kwh'] = np.log(filtered_data['total_kwh'])
    filtered_data['total_kwh_diff'] = filtered_data['total_kwh'].diff().fillna(0)
  
    
    # Split train/test
    df_train =  filtered_data[ 0 : int(0.8 * filtered_data.shape[0]) ] 
    df_test =   filtered_data[  int(0.8 * filtered_data.shape[0]) :  ] 
    
    return df_train, df_test


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
    return np.mean(np.abs((true_values - predicted_values) / true_values))

mrc_sector_list = data['sector_mrc'].unique()

# Préparer une liste pour stocker les résultats (MAPE, valeurs réelles, prédictions)
results = []
all_predictions = {}

# Boucle sur chaque combinaison de MRC et secteur
for mrc in mrc_sector_list:
    try:
        print(f"Entraînement pour MRC_Secteur: {mrc}")
        df_train, df_test = preprocess_data(data, mrc_sector=mrc)
        

        
        # Créer des générateurs
        train_generator = Generateur(df_train, batch_size=4)
        test_generator = Generateur(df_test, batch_size=4)
        
        # Réinitialiser le modèle avant chaque entraînement
        model = create_model(input_shape=(11, df_train.shape[1]), output_shape=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )
        
        # Entraîner le modèle
        history = model.fit(
            x=train_generator,
            epochs=50,
            validation_data=test_generator,
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
        plt.plot(epochs, history_dict['mean_absolute_error'], 'bo-', label='MAE d\'entraînement')
        plt.plot(epochs, history_dict['val_mean_absolute_error'], 'ro-', label='MAE de validation')
        plt.title(f'Évolution de l\'erreur absolue moyenne (MAE) - {mrc}')
        plt.xlabel('Époques')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        # Afficher les graphes
        plt.tight_layout()
        plt.savefig(f"Resultats_Training/training_metrics_{mrc}.png", format='png')
        plt.show()
        
        



        # 7.1. Générer des prédictions sur l'ensemble de test
        prediction_generator = Generateur(pd.concat([df_train, df_test], axis=0), batch_size=1, window_size=11)  
        predictions = model.predict(prediction_generator)
        predictions = np.exp(predictions)  # Revenir à la valeur d'origine 
                
        # Récupérer les vraies valeurs de `y` pour tout le dataset
        true_values_full = np.exp(prediction_generator.y)
        mape = calculate_mape(true_values_full, predictions)

        # Sauvegarder les prédictions et les vraies valeurs dans le dictionnaire
        all_predictions[f"{mrc}"] = {
            "true_values": true_values_full.flatten(),
            "predictions": predictions.flatten(),
            "mape": mape
        }    

    except ValueError as e:
        print(f"Erreur pour {mrc}: {e}")





#%%

for mrc in mrc_list:
    try:
        # Accéder aux prédictions et vraies valeurs depuis all_predictions
        predictions = all_predictions[f"{mrc}"]['predictions']
        true_values = all_predictions[f"{mrc}"]['true_values']
        
        # Ajouter les résultats dans la liste
        for i in range(len(true_values)):
            results.append({
                "Date": i,
                "MRC_Secteur": mrc,
                "Valeurs Réelles": true_values[i],
                "Prédictions": predictions[i]
            })
    except KeyError as e:
        print(f"Erreur pour {mrc}: {e}")
        
# Convertir la liste en DataFrame
df_results = pd.DataFrame(results)

# 2. Créer le graphique interactif avec Plotly
fig = px.line(
    df_results, 
    x="Date", 
    y=["Valeurs Réelles", "Prédictions"], 
    color="Secteur", 
    facet_col="MRC",  # Séparer les graphes par MRC
    title="Comparaison des Prédictions et des Valeurs Réelles par MRC et Secteur",
    labels={"value": "Consommation (kWh)", "variable": "Série"}
)

df_results.to_csv("Resultats_Training/score.csv")

# 3. Personnaliser les lignes des prédictions
fig.update_traces(
    selector=dict(name="Prédictions"), 
    line=dict(dash="dot")
)

# 4. Afficher le graphique interactif
fig.show()

#%%