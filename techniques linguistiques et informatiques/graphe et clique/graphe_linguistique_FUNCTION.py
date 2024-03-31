import networkx as nx

def créer_reseau_récursif(model, word, G, niveau, max_niveau=2):
    if niveau > max_niveau:
        return
    
    # Ajouter le nœud actuel au graphe
    G.add_node(word)
    
    # Obtenir les mots similaires
    similar_words = model.wv.most_similar(word, topn=10)
    
    # Ajouter les nœuds et les arêtes pour chaque mot similaire
    for similar_word, similarity_score in similar_words:
        G.add_node(similar_word)
        G.add_edge(word, similar_word, weight=similarity_score)
        
        # Appeler récursivement pour les mots similaires
        créer_reseau_récursif(model, similar_word, G, niveau+1, max_niveau)

def dessin_graphe(G):
    # Utiliser une disposition fruchterman_reingold pour une meilleure répartition des nœuds
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    
    # Dessiner les nœuds avec des paramètres pour améliorer la lisibilité
    nx.draw(G, pos, with_labels=False, node_size=30, node_color='skyblue')

     # Dessiner les arêtes avec une couleur plus claire et une transparence réduite
    nx.draw(G, pos, with_labels=False, edge_color='gray', alpha=1)
    
    # Dessiner les labels des nœuds de manière à éviter le chevauchement
    labels = nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', font_family='sans-serif', alpha=0.7)

