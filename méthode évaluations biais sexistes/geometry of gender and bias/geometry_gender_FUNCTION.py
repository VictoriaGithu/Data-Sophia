import numpy as np
from sklearn import decomposition

def diff_genre(model,a,b):
    model.unit_normalize_all()
    vect_a = model.get_vector(a)
    vect_b = model.get_vector(b)
    return (vect_a - vect_b)/np.linalg.norm(vect_a - vect_b)

def dir_genre(model, arr_diff):
    #arr_diff est un array qui contient toutes les différences de vecteurs 
    model.unit_normalize_all()
    D = np.transpose(arr_diff)
    pca = decomposition.PCA(n_components=9)
    pcs = pca.components_
    dir_genre = np.dot(D, pcs[0])
    return dir_genre/np.linalg.norm(dir_genre)

def biais_indirect(vect1, vect2, dir): 
    #dir est la dir du sous-espace associé au genre
    vect1_para = np.dot(vect1, dir)*dir
    vect1_perp = vect1 - vect1_para
    vect2_para = np.dot(vect2, dir)*dir
    vect2_perp = vect2 - vect2_para
    num_beta = np.dot(vect1_perp, vect2_perp)
    den_beta = (np.linalg.norm(vect1_perp)**2)*(np.linalg.norm(vect2_perp)**2)
    beta = (np.dot(vect1, vect2) - num_beta/den_beta) / np.dot(vect1, vect2)
    return beta  