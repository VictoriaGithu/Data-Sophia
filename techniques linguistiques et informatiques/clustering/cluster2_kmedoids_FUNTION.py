import numpy as np

# Fonction pour extraire les phrases contenant des co-occurrences de deux mots dans un corpus
def extract_phrases_with_co_occurrences(corpus, word1, word2):
    sentences = corpus.split('.')  # Divisez le corpus en phrases
    phrases_with_co_occurrences = []
    for sentence in sentences:
        words = sentence.strip().split()  # Divisez la phrase en mots
        if word1 in words and word2 in words:
            phrases_with_co_occurrences.append(sentence.strip())
    return phrases_with_co_occurrences

# Obtention des vecteurs de texte en moyennant les vecteurs de mots
def moyenne_vecteurs(mots, model):
    # Retourne le vecteur moyen pour une liste de mots
    vecteurs = [model.wv.get_vector(mot) for mot in mots if model.wv.has_index_for(mot)]
    return np.mean(vecteurs, axis=0)


from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess

def text_to_corpus_mots(text):
    list_mots = word_tokenize(text)
    list_mots_flat = []
    for word in list_mots:
        word_list = simple_preprocess(word, deacc=True)
        list_mots_flat.extend(word_list)
    return list_mots_flat



#On utilise l'algo de Kmedoids pour faire du clustering
from sklearn_extra.cluster import KMedoids

def kmedoids(n, phrases, model):
    vecteur_texte = []
    for phrase in phrases: 
        psg = text_to_corpus_mots(phrase) 
        vecteur_texte.append(moyenne_vecteurs(psg, model))
    # matrice de données ATTENTION : chaque ligne représente un vecteur)
    data_matrix = np.array(vecteur_texte)
    kmedoids = KMedoids(n_clusters=n, random_state=0, metric = 'cosine')
    clusters = kmedoids.fit_predict(data_matrix)
    return clusters, kmedoids, data_matrix

def reveal_cluster(clusters, kmedoids, phrases):
    # Trouver l'indice du medoid de chaque cluster
    medoid_indices = kmedoids.medoid_indices_
    
    # Extraire les vecteurs de chaque medoid
    #medoid_vectors = data_matrix[medoid_indices]
    
    # Convertir les vecteurs en phrases ou documents originaux
    medoid_phrases = []
    for index in medoid_indices:
        medoid_phrases.append(phrases[index])
    
    return clusters, medoid_phrases

