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


# calcul du degré moyen
degrees = resultat['degrees']
avg_degree = resultat['avg_degree']
print("Degré moyen du graphe :", avg_degree)

# traçage de l'histogramme des degrés
plt.hist(degrees, bins=range(max(degrees)+2), align='left', rwidth=0.8)
plt.xlabel('Degré')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des degrés des nœuds')
plt.show()

# calcul du moyen de degre de clusterisation
clustering_coeffs = resultat['clustering']
avg_clustering_coeff = resultat['avg_clustering']
print("Coefficient de clustering moyen du graphe :", avg_clustering_coeff)

# distribution des coefficients de clustering
plt.hist(list(clustering_coeffs.values()), bins=10, rwidth=0.8)
plt.xlabel('Coefficient de clustering')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des coefficients de clustering des nœuds')
plt.show()

# calcul du nombre des cliques et leur ordre
cliques = resultat['cliques']
num_cliques = len(cliques)
clique_orders = [len(clique) for clique in cliques]
print("Nombre de cliques dans le graphe :", num_cliques)
print("Ordres des cliques :", clique_orders)

# nombre des composantes connexes et leur ordre
connected_components = resultat['connected_components']
num_connected_components = len(connected_components)
connected_component_orders = [len(component) for component in connected_components]
print("Nombre de composantes connexes dans le graphe :", num_connected_components)
print("Ordres des composantes connexes :", connected_component_orders)

# longeur des chemins les plus courts entre toutes les paires de nœuds connectés
lengths = resultat['unweighted_shortest_path_lengths']
print("Longueurs des plus courts chemins entre toutes les paires de nœuds connectés :", lengths)

# distribution des plus courts chemins
plt.hist(lengths, bins=10, rwidth=0.8)
plt.xlabel('Longueur des plus courts chemins')
plt.ylabel('Nombre de paires de nœuds')
plt.title('Histogramme des longueurs des plus courts chemins entre les paires de nœuds')
plt.show()








# Partie 3:
# creer un graphe pondéré par la distance euclidienne au carré entre les nœuds connectés pour un d_max de 60000

G_weighted = build_graph(positions, d_max, is_weighted=True)





plot_graph_3d(G_weighted)
plt.show() # Affichage du graphe 3D


# recupération des métriques du graphe pondéré
resultat_weighted = calcul_resultat(G_weighted, is_weighted=True)

# calcul du degré moyen
degrees_weighted = resultat_weighted['degrees']
avg_degree_weighted = resultat_weighted['avg_degree']
print("Degré moyen du graphe pondéré :", avg_degree_weighted)

# traçage de l'histogramme des degrés
plt.hist(degrees_weighted, bins=range(max(degrees_weighted)+2), align='left', rwidth=0.8)
plt.xlabel('Degré')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des degrés des nœuds du graphe pondéré')
plt.show()


# calcul du moyen de degre de clusterisation
clustering_coeffs_weighted = resultat_weighted['clustering']
avg_clustering_coeff_weighted = resultat_weighted['avg_clustering']
print("Coefficient de clustering moyen du graphe pondéré :", avg_clustering_coeff_weighted)


# distribution des coefficients de clustering
plt.hist(list(clustering_coeffs_weighted.values()), bins=10, rwidth=0.8)
plt.xlabel('Coefficient de clustering')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des coefficients de clustering des nœuds du graphe pondéré')
plt.show()


# calcul du nombre des cliques et leur ordre
cliques_weighted = resultat_weighted['cliques']
num_cliques_weighted = len(cliques_weighted)
clique_orders_weighted = [len(clique) for clique in cliques_weighted]
print("Nombre de cliques dans le graphe pondéré :", num_cliques_weighted)
print("Ordres des cliques du graphe pondéré :", clique_orders_weighted)


# nombre des composantes connexes et leur ordre
connected_components_weighted = resultat_weighted['connected_components']
num_connected_components_weighted = len(connected_components_weighted)
connected_component_orders_weighted = [len(component) for component in connected_components_weighted]
print("Nombre de composantes connexes dans le graphe pondéré :", num_connected_components_weighted)
print("Ordres des composantes connexes du graphe pondéré :", connected_component_orders_weighted)


# longeur des chemins les plus courts entre toutes les paires de nœuds
lengths_weighted = resultat_weighted['weighted_shortest_path_lengths']
print("Longueurs des plus courts chemins entre toutes les paires de nœuds connectés du graphe pondéré :", lengths_weighted)


# distribution des plus courts chemins
plt.hist(lengths_weighted, bins=10, rwidth=0.8)
plt.xlabel('Longueur des plus courts chemins')
plt.ylabel('Nombre de paires de nœuds')
plt.title('Histogramme des longueurs des plus courts chemins entre les paires de nœuds du graphe pondéré')
plt.show()


