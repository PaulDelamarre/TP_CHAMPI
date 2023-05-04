
# TP_CHAMPI

# Structure du GITHUB
Au cours de ce projet, nous avons décidé de **produire un modèle de reconnaissance automatique sur le TOP 10 des Champignons français accepté sur le site *http://mushroomobserver.org/* avec une confiance supérieure à 0.95.** 

Pour récupérer le projet, il suffit dans un terminal de  :

    git clone https://github.com/PaulDelamarre/TP_CHAMPI.git

Dans notre cas nous avons réalisé **4 notebooks** :

 - **1_ExplorationDonnées_ChoixProblématique.ipynb**

Au cours de ce notebook, nous importons les données JSON. Nous définissons notre problématique de projet. En sortie, nous obtenons 2 dataframes (1 au format .csv - 1 obtenu a chaque fois en relançant une fonction car pas possible d'exporter en .csv et .pickle a cause de la taille)

 - **2_Etude_RGB&TailleImage.ipynb**
 
Dans ce notebook, nous réalisons plusieurs études pour découvrir notre dataset d'étude (notamment sur les tailles d'images, sur les couleurs RGB ...)

 - **3_deeplearning1.ipynb**
 
Au cours de ce notebook, nous construisons plusieurs réseaux de neurones à la main et comparons les résultats.

 - **4_deeplearning2-resnet.ipynb**
 
 Dans ce notebook, nous utilisons le réseau de neurones pré entrainé (ResNet34) pour le fine tuner sur notre dataset. Nous produisons également des sorties pour réaliser un tableau de bord sous Power BI.

Le tableau de bord n'a pas pu être publié car il est trop volumineux, il sera présenté le jour de la soutenance. Tout notre projet est présenté en détail dans le rapport rendu sur Moodle.

# Règle générale :

Pour l'ensemble des sujets proposés, il faudra présenter :

- Une exploration des données bien construite, avec à l'appui une description des variables, leurs relations, les points remarquables.

- Il faudra que cette exploration soit appuyée de visualisations soignées et pertinentes. (Incluez au moins une visualisation avec plotly, une bibliothèque en vogue pour la visualisation des données.)

- N'hésitez pas à procéder à une phase de "feature engineering" (si possible) --> création / sélection de variables d'intérêt.

- Pourquoi pas faire évoluer ou approfondir la problématique.

- Des cycles de modélisation : explorez plusieurs modèles possibles.

- Au moins un cycle d'optimisation.

- Une modélisation finale avec scoring complet du/des modèles retenus.

- Optionnel : pourquoi pas s'intéresser à l'interprétabilité du modèle ?

- Une analyse des résultats obtenus.

Idéalement, vous mettrez en place un repository github pour me proposer un rendu avec :

- Un rapport de projet qui doit contenir :
  - Une présentation des données, leur contexte et la position du problème.
  - Un descriptif de la méthodologie choisie et mise en place, des différentes étapes (et aussi des modèles choisis).
  - Une description des résultats obtenus (y compris de la phase d'optimisation).
  - Une critique des résultats.
  - Les problèmes rencontrés et comment vous y avez fait face.
  - Des perspectives (que feriez-vous en plus).
  - Une conclusion.
- Les codes (langage python) du projet (et aussi de la visualisation éventuellement).

La présentation d'une durée de 25 à 30 minutes devra reprendre les étapes principales de votre rapport. Vous êtes libres concernant le support de présentation. Les supports interactifs (de type streamlit ou dash) peuvent être envisagés. Tout le monde doit prendre la parole.

Je sais :

- Que les projets existent sur internet... ne les reproduisez pas à l'identique, ce serait une (très) mauvaise idée !

- Qu'il y a différents niveaux de difficultés → je le prendrai en compte lors de l’évaluation du projet.

L'ensemble codes + rapport est attendu au plus tard le 01 Mai. La présentation aura lieu le 09 Mai.

----

# 4ème sujet : Reconnaissance automatique de champignons

Challenge ? Le projet consiste à créer un modèle de reconnaissance automatique de champignons.

Voici une adresse : [https://github.com/bechtle/mushroomobser-dataset](https://github.com/bechtle/mushroomobser-dataset).

Vous y trouverez des informations sur des images de champignons. Dans le « Read me », vous
avez un lien pour télécharger un set de données qui se compose d’images et de fichiers JSON.
Dans ce projet, vous devrez absolument choisir et poursuivre un axe de recherche. En effet,
à priori, vous ne pourrez pas réaliser des prédictions solides pour toutes images de
champignons disponibles. Il faudra notamment choisir l’échelle à laquelle vous souhaitez
travailler (échelle dans la classification phylogénique : espèce, famille, genre………).


