
import networkx as nx

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open('assets/topology_high.csv', 'r')
lignes = f.readlines()  # Retourne une liste
f.close()
lignes = lignes[1:]  # On enlève l'en-tête
G = nx.Graph()
nb_nodes = len(lignes)
for i in range(nb_nodes):
   li = (lignes[i].split(','))[1:]
   p = list(map(float, li)) 
   G.add_node(i, pos=p)




d_max = 20000

# Création du graphe



    
# Ajout des arêtes selon la distance
for i in range(nb_nodes):
    for j in range(i+1, nb_nodes):
        
        d = np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))
        if d <= d_max:
            G.add_edge(i, j, weight=d)

print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())



fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')

pos = nx.get_node_attributes(G, 'pos')

# Nœuds
xs = [pos[i][0] for i in G.nodes()]
ys = [pos[i][1] for i in G.nodes()]
zs = [pos[i][2] for i in G.nodes()]
ax.scatter(xs, ys, zs, color='red')

# Arêtes
for i, j in G.edges():
    x = [pos[i][0], pos[j][0]]
    y = [pos[i][1], pos[j][1]]
    z = [pos[i][2], pos[j][2]]
    ax.plot(x, y, z, color='blue')


plt.show()
"""

G = nx.cubical_graph()
subax1 = plt.subplot(121)
nx.draw(G)   # default spring_layout
subax2 = plt.subplot(122)
nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')

"""