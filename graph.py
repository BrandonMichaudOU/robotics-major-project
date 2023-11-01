class Node:
    def __init__(self, id):
        self.id = id
        self.adjacent = {} # dictionary from node to weight

    def add_neighbor(self, neighbor, weight=1):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]
    
    def get_id(self):
        return self.id

class Graph:
    def __init__(self):
        self.nodes = [] # List of nodes in graph
        self.num_nodes = 0

    def add_node(self, id):
        self.num_nodes = self.num_nodes + 1
        new_node = Node(id)
        self.nodes.append(new_node)
        return new_node

    def add_edge(self, source, dest, weight=1):
        if source not in self.nodes:
            self.add_node(source)
        if dest not in self.nodes:
            self.add_node(dest)

        source.add_neighbor(dest, weight)
        dest.add_neighbor(source, weight)

    def get_nodes(self):
        return self.nodes
    
if __name__ == '__main__':
    g = Graph()

    n1 = g.add_node(1)
    n2 = g.add_node(2)
    n3 = g.add_node(3)

    g.add_edge(n1, n2, 4)
    g.add_edge(n2, n3)
    g.add_edge(n3, n1, 9)

    for node in g.get_nodes():
        for neighbor in node.get_connections():
            print(f'({node.get_id()}, {neighbor.get_id()}): {node.get_weight(neighbor)}')