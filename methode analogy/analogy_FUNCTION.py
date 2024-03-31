import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


def mat_etape_analogy(model, a, b): 
#il faut normaliser les vecteurs du modèle
    model.unit_normalize_all()
    X = model.vectors

#on trouve les vecteurs associés aux mots 
    vect_a = model.get_vector(a)
    vect_b = model.get_vector(b)
    diff_visée = (vect_a - vect_b)/np.linalg.norm(vect_a - vect_b)
#me donne une ligne et 24065 colonnes
#chaque colonne correspond à la cos_sim entre un mot et le vect diff she-he

    mat_1 = linear_kernel((diff_visée).reshape(1,-1), X) 
    nb_words = len(model.index_to_key)
    
#matrice donne les similarités cos entre le vecteur diff et tous les autres vecteurs
    sim_mat = np.zeros((nb_words,nb_words))
    xmat_1 = np.array(mat_1[0,:])
    sim_mat = np.tile(xmat_1[np.newaxis,:], (nb_words,1))-np.tile(xmat_1[:,np.newaxis], (1,nb_words))
    


def trouver_analogy(model, sim_mat, n):
# Supposons que sim_mat soit votre matrice de similarité et model votre array de vecteurs
    best_coefs = -np.ones(n)
    indices_best = [0] * n
    k = 10*n
    while np.min(best_coefs) == -1: 
# Utilisez argpartition pour obtenir les indices triés des 1000 plus grands coefficients de toute la matrice
        sorted_indices = np.argpartition(-sim_mat, 10**k, axis=None)[:10**k]

# Utilisez unravel_index pour obtenir les indices dans la matrice d'origine
        indices_in_sim_mat = np.unravel_index(sorted_indices, sim_mat.shape)

# Parcourez les indices et mettez à jour les listes best_coefs et indices_best
        for i, j in zip(*indices_in_sim_mat):
            if sim_mat[i, j] > np.min(best_coefs):
                if np.linalg.norm(model[i] - model[j]) < 1:
                    index = np.argmin(best_coefs)
                best_coefs[index] = sim_mat[i, j]
                indices_best[index] = (i, j)
        k += 1 
    return indices_best, best_coefs

def reveal_analogy(model, indices_best, best_coefs):
    for i,j in indices_best :
        analog_she = model.index_to_key[i]
        analog_he = model.index_to_key[j]
    return (analog_she, analog_he)


