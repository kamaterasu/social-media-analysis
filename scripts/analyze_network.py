import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# Load data
data = pd.read_csv("../data/edges.csv")
graph = nx.from_pandas_edgelist(data, source='Source', target='Target')

# Visualize the graph
nx.draw(graph, with_labels=True, node_color='skyblue',
        node_size=1500, edge_color='gray')
plt.show()

# Өгөгдөл ачаалах
өгөгдөл = pd.read_csv("../data/edges.csv")
график = nx.from_pandas_edgelist(өгөгдөл, эх сурвалж='Эх сурвалж', зорилтот='Зорилтот')

# Графикийг дүрслэн харуулах
nx.draw(график, шошготой=Үнэн, node_color='skyblue',
        node_size=1500, edge_color='саарал')
plt.show()
