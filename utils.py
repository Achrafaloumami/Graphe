import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Lecture du fichier CSV et création les positions des nœuds
def read_positions(filepath):
    """Lit un fichier CSV (id,x,y,z) et renvoie une liste de positions numpy.
    Retourne: list[np.array([x,y,z])]
    """
    f = open(filepath, 'r')
    lignes = f.readlines()  # Retourne une liste de lignes
    f.close()
    lignes = lignes[1:]  # Suppression de la première ligne (en-têtes)
    positions = []
    for li in lignes:
        parts = li.strip().split(',')
        coords = list(map(float, parts[1:]))
        positions.append(np.array(coords))
    return positions


# Création du graphe
def build_graph(positions, d_max, is_weighted):
    
    G = nx.Graph()
    nb_nodes = len(positions)

    for i in range(nb_nodes):
        G.add_node(i, pos=positions[i])

    for i in range(nb_nodes):
        for j in range(i+1, nb_nodes):
            d = np.linalg.norm(G.nodes[i]['pos'] - G.nodes[j]['pos'])  # Calcul de la distance euclidienne entre les nœuds i et j
            if d <= d_max:
                if is_weighted:
                    w = d**2    # Poids de l'arête (distance au carré)
                    G.add_edge(i, j, weight=w)
                else:
                    G.add_edge(i, j) # Ajout de l'arête des poids implicites égaux à 1
    return G


# Tracé du graphe 3D
def plot_graph_3d(G , node_color='red', edge_color='blue', node_size=20, alpha=1.0):


    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') # Création du subplot 3D

    pos = nx.get_node_attributes(G, 'pos')
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]
    zs = [pos[i][2] for i in G.nodes()]
    ax.scatter(xs, ys, zs, color=node_color, s=node_size)

    for i, j in G.edges():
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        z = [pos[i][2], pos[j][2]]
        ax.plot(x, y, z, color=edge_color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


# Calcul des métriques du graphe
def calcul_resultat(G, is_weighted):

    # Dictionnaire pour stocker les métriques
    resultat = {}
    
    nb_nodes = G.number_of_nodes()
    # Calcul du degré moyen
    degrees = [d for (_, d) in G.degree()]
    avg_degree = sum(degrees) / nb_nodes

    # degré de clusterisation
    clustering = nx.clustering(G)

    # coefficient de clusterisation moyen
    avg_clustering = nx.average_clustering(G)

    # cliques 
    cliques = list(nx.find_cliques(G))

    # composantes connexes
    connected_components = list(nx.connected_components(G))


    # longueurs des chemins les plus courts entre toutes les paires de nœuds connectés    
    if is_weighted:
        short_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    else:
        short_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    lengths = []
    for source in short_path_lengths:
        for target in short_path_lengths[source]:
            if source != target:
                lengths.append(short_path_lengths[source][target])

    # Remplissage du dictionnaire des métriques par les valeurs calculées
    resultat['degrees'] = degrees
    resultat['avg_degree'] = avg_degree
    resultat['clustering'] = clustering
    resultat['avg_clustering'] = avg_clustering
    resultat['cliques'] = cliques
    resultat['connected_components'] = connected_components

    if is_weighted:
        resultat['weighted_shortest_path_lengths'] = lengths
    else:
        resultat['unweighted_shortest_path_lengths'] = lengths
    return resultat