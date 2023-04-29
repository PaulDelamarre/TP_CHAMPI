# Importation
import numpy as np              # linear algebra
import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)
import os                       # files handling
import re
from PIL import Image
from random import randint, seed
from IPython.display import display
import matplotlib.pyplot as plt
import json
import warnings
import requests
import seaborn as sns
warnings.filterwarnings('ignore')
import cv2
import tensorflow
from tensorflow import keras
from keras.models import Sequential # Pour construire un réseau de neurones
from keras.layers import Dense, Dropout, Flatten, LeakyReLU # Pour instancier une couche dense
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


# Fonction pour charger toutes les images

def load_jpeg_images(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.jpeg')]
    images = []
    ids = []
    for file in image_files:
        ids.append(os.path.splitext(file)[0])
        with Image.open(os.path.join(path, file)) as image:
            image_data = np.array(image)
            img = cv2.resize(image_data, (100, 100))
            images.append(img)

    dataimage = pd.DataFrame({"ids": ids, "img": images})
    datachampi = pd.read_csv('C:/Users/baptiste/Documents/Etude/CNAM/Année2/Semestre 2/Fouille de données/Projet/TP_CHAMPI/dataframe/champignons.csv')

    datachampi = datachampi[['label', 'image_id']]

    dataimage[['ids']] = dataimage[['ids']].astype('int64')

    dataset = pd.merge(dataimage, datachampi, left_on='ids', right_on='image_id', how='left')

    return dataset


df = load_jpeg_images("C:/Users/baptiste/Documents/Etude/CNAM/Année2/Semestre 2/Fouille de données/Projet/TP_CHAMPI/images")
df = df.drop(index=8431)
# Mise en place du jeu de X et Y en numpy pour qu'il puisse être traité dans les réseaux de neuronnes
display(df.head())
X = df['img'].to_numpy()
Y = df['label'].to_numpy()
for i in range(len(X)):
    if str(X[i].shape) != "(100, 100, 3)":
        print("Yes"+str(X[i].shape)+str(i))

# X = [np.array(x) for x in X]

# Modification de la taille de X pour que tous ces éléments soient pris en compte dans ses dimensions
X = np.stack(X)
print(X.shape)
print(Y.shape)

# Redéfinition des données

# One hot encoding
Y = pd.get_dummies(Y)

# Création du jeu d'entrainement et du jeu de test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25)
print('Les dimensions de X_train est de '+str(X_train.shape)+'.')
print('Les dimensions de Y_train est de '+str(Y_train.shape)+'.')
print('Les dimensions de X_test est de '+str(X_test.shape)+'.')
print('Les dimensions de Y_test est de '+str(Y_test.shape)+'.')

# Normaliser
X_train2 = X_train/255
X_test2 = X_test/255

nb_class = Y_test.shape[1]
print('Le nombre de classe du jeu de donnée est ',str(nb_class))

# Définir la fonction qui construit votre modèle
def build_model(filters, kernel_size, padding, activation, dropout_rate, dense_units, leaky_alpha):
    model = Sequential()
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                     input_shape=(100, 100, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=100, activation=LeakyReLU(alpha=leaky_alpha)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=nb_class, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model, epochs=50, batch_size=32, verbose=0)

# Définir les hyperparamètres à tester
filters = [16, 32, 64]
kernel_size = [(3, 3), (5, 5), (7, 7)]
padding = ['valid', 'same']
activation = ['relu', 'sigmoid']
dropout_rate = [0.5, 0.7]
dense_units = [50, 100, 200]
leaky_alpha = [0.1, 0.2, 0.3]

# Créer un dictionnaire des hyperparamètres
param_grid = dict(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                  dropout_rate=dropout_rate, dense_units=dense_units, leaky_alpha=leaky_alpha)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_test2, Y_test)

# Afficher les meilleurs hyperparamètres trouvés
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Afficher les résultats pour toutes les combinaisons d'hyperparamètres testées
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))