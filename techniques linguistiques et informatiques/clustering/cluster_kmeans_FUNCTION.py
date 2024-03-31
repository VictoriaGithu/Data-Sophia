# Maintenant, on veut comprendre les versets responsables des rapprochements de ce graphe. 

# Pour ce faire, on va se focus sur un noeud à chaque fois et extraire les versets où il y a occurence d'un ou plusieurs des termes concernés. On va ensuite faire du clustering pour comprendre mieux ce corpus particulier. 

from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
import numpy as np
import nltk
nltk.download('punkt')

def text_to_corpus_mots(text):
    list_mots = word_tokenize(text)
    list_mots_flat = []
    for word in list_mots:
        word_list = simple_preprocess(word, deacc=True)
        list_mots_flat.extend(word_list)
    return list_mots_flat

def passages_cooccurrence(corpus, mots_etudies, fenetre_taille = 5, taille_versets = 10):
    # Tokenisation en mots
    mots = text_to_corpus_mots(corpus)   
    passages_cooccurrence = []
    for i in range(len(mots) - 2*fenetre_taille):
        mots_fenetre = mots[i:i+2*fenetre_taille]
        if len(set(mots_etudies).intersection(set(mots_fenetre))) >= 2:
            verset = mots[i-taille_versets:i+taille_versets]
            passages_cooccurrence.append(verset)
    return passages_cooccurrence

def afficher_passages_coocurrences(passages_cooccurrence, mots_etudies, fenetre_taille = 5):
    print(f"Passages avec co-occurrence d'au moins 2 mots parmi {mots_etudies} dans une fenêtre de {fenetre_taille} mots :")
    for passage in passages_cooccurrence:
        print(passage)
        print('')




# Obtention des vecteurs de texte en moyennant les vecteurs de mots
def moyenne_vecteurs(mots, model):
    # Retourne le vecteur moyen pour une liste de mots
    vecteurs = [model.wv.get_vector(mot) for mot in mots if model.wv.has_index_for(mot)]
    return np.mean(vecteurs, axis=0)

import torch
from torch_kmeans import KMeans

#AVEC KMEANS

def k_means(n, passages_cooccurrence, model):
    vecteur_texte = []
    for passage in passages_cooccurrence:
        vecteur_texte.append(moyenne_vecteurs(passage, model))
    # matrice de données ATTENTION : chaque ligne représente un vecteur)
    data_matrix = np.array(vecteur_texte) 
    torch_tensor = torch.from_numpy(data_matrix)
    # Ajoutez une dimension supplémentaire pour la batch_size
    batch_size_one_tensor = torch_tensor.unsqueeze(0)
    # batch_size_one_tensor aura maintenant une batch_size de 1
    model = KMeans(n_clusters=n, BaseDistance='cosine')
    model = model.fit(batch_size_one_tensor)
    labels = model.predict(batch_size_one_tensor)
    # Convertissez le tensor en un tableau NumPy
    labels_numpy = labels.numpy()
    return labels_numpy


def reveal_clusters(labels, passages_cooccurrence, n_clusters):
    #fonction avec print
    for i in range(n_clusters):
        indices = np.where(labels == i)[1]
        print(f"Cluster {i} :")
        for i in range(len(indices)):
            print(passages_cooccurrence[indices[i]])
        print('')


def reveal_clusters_2(labels, passages_cooccurrence, n_clusters):
    #fonction avec liste
    dict_list = {}
    
    for cluster_label in range(n_clusters):
        indices = np.where(labels == cluster_label)[1]
        dict_list[cluster_label] = []
        
        for index in range(len(indices)):
            dict_list[cluster_label].append(passages_cooccurrence[indices[index]])
            
    return dict_list

def tri_reveal(list_i):
    # la liste i est la liste des phrases associées au cluster i 
    j = 0 
    k = 0
    new_list = []
    while len(set(list_i[j]).intersection(set(list_i[j+1]))) > 38:
        j += 1
        k += 1
    new_member = set(list_i[j]).union(set(list_i[j+k]))
    new_list.append(new_member)
    return new_list