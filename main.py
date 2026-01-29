
import networkx as nx

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Lecture du fichier CSV et création des nœuds
f = open('assets/topology_high.csv', 'r')
lignes = f.readlines()  # Retourne une liste de lignes
f.close()
lignes = lignes[1:]  # Suppression de la première ligne (en-têtes)
G = nx.Graph() # Création du graphe vide
nb_nodes = len(lignes)
for i in range(nb_nodes):
   li = (lignes[i].split(','))[1:]
   p = np.array(list((map(float, li)))) # Conversion des coordonnées en float
   G.add_node(i, pos=p)




d_max = 20000 # Distance maximale de connexion entre deux nœuds

# Création du graphe



    
# Ajout des arêtes selon la distance
for i in range(nb_nodes):
    for j in range(i+1, nb_nodes):
        
        d = np.linalg.norm(G.nodes[i]['pos'] - G.nodes[j]['pos']) # Calcul de la distance euclidienne entre les nœuds i et j
        if d <= d_max:
            d = d**2
            G.add_edge(i, j, weight=d) # Ajout de l'arête si la distance est inférieure à d_max

print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())



fig = plt.figure() # Création de la figure


ax = fig.add_subplot(111, projection='3d') # Création du subplot 3D

pos = nx.get_node_attributes(G, 'pos') # Récupération des positions des nœuds

# Nœuds
xs = []
ys = []
zs = []
for i in G.nodes():
    xs.append(pos[i][0])
    ys.append(pos[i][1])
    zs.append(pos[i][2])

ax.scatter(xs, ys, zs, color='red') # Tracé des nœuds

# Arêtes
for i, j in G.edges():
    x = [pos[i][0], pos[j][0]]
    y = [pos[i][1], pos[j][1]]
    z = [pos[i][2], pos[j][2]]
    ax.plot(x, y, z, color='blue') # Tracé des arêtes


plt.show() # Affichage du graphe 3D


"""

G = nx.cubical_graph()
subax1 = plt.subplot(121)
nx.draw(G)   # default spring_layout
subax2 = plt.subplot(122)
nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')

"""


# calcul du degré moyen
degrees = [val for ( _ , val) in G.degree()]
avg_degree = sum(degrees) / nb_nodes
print("Degré moyen du graphe :", avg_degree)

# traçage de l'histogramme des degrés
plt.hist(degrees, bins=range(max(degrees)+2), align='left', rwidth=0.8)
plt.xlabel('Degré')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des degrés des nœuds')
plt.show()

# calcul du moyen de degre de clusterisation
clustering_coeffs = nx.clustering(G)
avg_clustering_coeff = nx.average_clustering(G) #sum(clustering_coeffs.values()) / nb_nodes
print("Coefficient de clustering moyen du graphe :", avg_clustering_coeff)

# distribution des coefficients de clustering
plt.hist(list(clustering_coeffs.values()), bins=10, rwidth=0.8)
plt.xlabel('Coefficient de clustering')
plt.ylabel('Nombre de nœuds')
plt.title('Histogramme des coefficients de clustering des nœuds')
plt.show()

# calcul du nombre des cliques et leur ordre
cliques = list(nx.find_cliques(G))
num_cliques = len(cliques)
clique_orders = [len(clique) for clique in cliques]
print("Nombre de cliques dans le graphe :", num_cliques)
print("Ordres des cliques :", clique_orders)

# nombre des composantes connexes et leur ordre
connected_components = list(nx.connected_components(G))
num_connected_components = len(connected_components)
connected_component_orders = [len(component) for component in connected_components]
print("Nombre de composantes connexes dans le graphe :", num_connected_components)
print("Ordres des composantes connexes :", connected_component_orders)



# distrbution des plus courts chemins
path_lengths = dict(nx.all_pairs_shortest_path_length(G))
lengths = []
for source in path_lengths:
    for target in path_lengths[source]:
        if source != target:
            lengths.append(path_lengths[source][target])

plt.hist(lengths, bins=10, rwidth=0.8)
plt.xlabel('Longueur des plus courts chemins')
plt.ylabel('Nombre de paires de nœuds')
plt.title('Histogramme des longueurs des plus courts chemins entre les paires de nœuds')
plt.show()




distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))