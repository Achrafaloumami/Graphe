import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

from utils import calcul_resultat, read_positions, build_graph, plot_graph_3d


# Partie 1:


# Lecture du fichier CSV
positions = read_positions('assets/topology_high.csv')


d_max = 60000 # Distance maximale de connexion entre deux nœuds

# Création du graphe
# Ajout des arêtes selon la distance
G = build_graph(positions, d_max, is_weighted=False)

print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())

# fig = plt.figure() # Création de la figure
# ax = fig.add_subplot(111, projection='3d') # Création du subplot 3D

plot_graph_3d(G)
plt.show() # Affichage du graphe 3D


# Partie 2:

# recupération des métriques du graphe non pondéré
resultat = calcul_resultat(G, is_weighted=False)

# Créer une figure avec 4 subplots (2 lignes, 2 colonnes)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogramme des degrés
degrees = resultat['degrees']
avg_degree = resultat['avg_degree']
print("Degré moyen du graphe :", avg_degree)

axes[0, 0].hist(degrees, bins=range(max(degrees)+2), align='left', rwidth=0.8)
axes[0, 0].set_xlabel('Degré')
axes[0, 0].set_ylabel('Nombre de nœuds')
axes[0, 0].set_title('Histogramme des degrés des nœuds')

# 2. Histogramme des coefficients de clustering
clustering_coeffs = resultat['clustering']
avg_clustering_coeff = resultat['avg_clustering']
print("Coefficient de clustering moyen du graphe :", avg_clustering_coeff)

axes[0, 1].hist(list(clustering_coeffs.values()), bins=10, rwidth=0.8)
axes[0, 1].set_xlabel('Coefficient de clustering')
axes[0, 1].set_ylabel('Nombre de nœuds')
axes[0, 1].set_title('Histogramme des coefficients de clustering des nœuds')

# 3. Infos sur les cliques et composantes connexes
cliques = resultat['cliques']
num_cliques = len(cliques)
clique_orders = [len(clique) for clique in cliques]
print("Nombre de cliques dans le graphe :", num_cliques)
print("Ordres des cliques :", clique_orders)

connected_components = resultat['connected_components']
num_connected_components = len(connected_components)
connected_component_orders = [len(component) for component in connected_components]
print("Nombre de composantes connexes dans le graphe :", num_connected_components)
print("Ordres des composantes connexes :", connected_component_orders)

info_text = f"Nombre de cliques : {num_cliques}\nNombre de composantes : {num_connected_components}\nOrdres composantes : {connected_component_orders}"
axes[1, 0].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', family='monospace')
axes[1, 0].axis('off')
axes[1, 0].set_title('Cliques et Composantes connexes')

# 4. Histogramme des longueurs des plus courts chemins
lengths = resultat['unweighted_shortest_path_lengths']
print("Longueurs des plus courts chemins entre toutes les paires de nœuds connectés :", lengths)

axes[1, 1].hist(lengths, bins=10, rwidth=0.8)
axes[1, 1].set_xlabel('Longueur des plus courts chemins')
axes[1, 1].set_ylabel('Nombre de paires de nœuds')
axes[1, 1].set_title('Histogramme des longueurs des plus courts chemins')

plt.tight_layout()
plt.show()








# Partie 3:
# creer un graphe pondéré par la distance euclidienne au carré entre les nœuds connectés pour un d_max de 60000

G_weighted = build_graph(positions, d_max, is_weighted=True)





plot_graph_3d(G_weighted)
plt.show() # Affichage du graphe 3D


# recupération des métriques du graphe pondéré
resultat_weighted = calcul_resultat(G_weighted, is_weighted=True)



# longeur des chemins les plus courts entre toutes les paires de nœuds
lengths_weighted = resultat_weighted['weighted_shortest_path_lengths']
lengths_weighted = [float(length) for length in lengths_weighted]
print("Longueurs des plus courts chemins entre toutes les paires de nœuds connectés du graphe pondéré :", lengths_weighted)


# distribution des plus courts chemins
plt.hist(lengths_weighted, bins=10, rwidth=0.8)
plt.xlabel('Longueur des plus courts chemins')
plt.ylabel('Nombre de paires de nœuds')
plt.title('Histogramme des longueurs des plus courts chemins entre les paires de nœuds du graphe pondéré')
plt.show()


