# Description textuelle d'images

### Description du projet
La description textuelle d’images est un challenge de l’Intelligence Artificielle qui combine les domaines de la vision artificielle et du traitement du langage naturel. Elle consiste en la génération automatique de la description textuelle d’une image.

Dans ce projet, nous allons essayer d'implémenter un modèle capable de générer automatiquement des descriptions textuelles à partir d'une image donnée en entrée. Pour ce faire, nous avons choisi d'utiliser un modèle VGG16 préentraîné sur ImageNet pour extraires les features des images. En ce qui est de la génération de texte, nous utilisons des RNN.

Notre modèle a été entraîné sur la base de données [Flickr8K](https://www.kaggle.com/adityajn105/flickr8k?select=Images).

### Organisation du répertoire
Dans ce répertoire, vous trouverez :
- un Jupyter Notebook **va_project.ipynb** qui contient notre algorithme principal pour entraîner et évaluer notre modèle.
- un fichier python **utils.py** qui contient toutes les petites fonctions que nous écrites pour prétaiter nos données et entraîner notre modèle.
- un document csv **df_texts.csv** donc lequel nous avons enregistré les descriptions textuelles de notre base de données après les avoir prétraitées afin de ne pas avoir à les prétraiter tout le temps.
- un fichier texte **requirements.txt** contenant les librairies que nous avons utilisées.

### Instructions supplémentaires
Afin d'exécuter ce code, vous devez commencer par télécharger la base de données [Flickr8K](https://www.kaggle.com/adityajn105/flickr8k?select=Images), décompresser l'archive obtenue et la renommée en flickr8k. Il vous faut aussi installer les librairies contenues dans notre requirements.txt avec la commande :
```
pip install -r requirements.txt
```
