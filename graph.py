class Node:
    def __init__(self, pos):
        self.pos = pos
        self.adjacent = {} # dictionary from node to weight

    def add_neighbor(self, neighbor, weight=1):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]
    
    def get_pos(self):
        return self.pos

class Graph:
    def __init__(self):
        self.nodes = [] # List of nodes in graph
        self.num_nodes = 0

    def add_node(self, pos):
        self.num_nodes = self.num_nodes + 1
        new_node = Node(pos)
        self.nodes.append(new_node)
        return new_node

    def add_edge(self, source, dest, weight=1):
        if source not in self.nodes:
            self.add_node(source)
        if dest not in self.nodes:
            self.add_node(dest)

        source.add_neighbor(dest, weight)
        dest.add_neighbor(source, weight)

    def get_connections(self, source):
        return source.adjacent.keys()  

    def get_nodes(self):
        return self.nodes
    
class SearchNode:
    def __init__(self, node, parent):
        self.node = node
        self.parent = parent

def breadth_first_search(graph, start, end):
    q = []
    seen = []
    root = SearchNode(start, None)
    q.append(root)
    seen.append(start)
    while len(q) != 0:
        n = q.pop(0)
        if n.node == end:
            return n
        for neighbor in graph.get_connections(n.node):
            if neighbor not in seen:
                seen.append(neighbor)
                next = SearchNode(neighbor, n)
                q.append(next)

    
if __name__ == '__main__':
    g = Graph()

    n1 = g.add_node((0, 0))
    n2 = g.add_node((5, 2))
    n3 = g.add_node((5, -2))
    n4 = g.add_node((8, 0))
    n5 = g.add_node((10, 1))
    n6 = g.add_node((12, 2))
    n7 = g.add_node((12, 0))
    n8 = g.add_node((15, 0))
    n9 = g.add_node((20, 0))
    n10 = g.add_node((20, -2))

    g.add_edge(n1, n2, 2)
    g.add_edge(n1, n3, 5)
    g.add_edge(n2, n3, 1)
    g.add_edge(n2, n4, 3)
    g.add_edge(n2, n5, 1)
    g.add_edge(n3, n4, 2)
    g.add_edge(n3, n7, 6)
    g.add_edge(n3, n10, 30)
    g.add_edge(n4, n5, 1)
    g.add_edge(n4, n7, 8)
    g.add_edge(n5, n6, 7)
    g.add_edge(n6, n7, 1)
    g.add_edge(n6, n8, 6)
    g.add_edge(n7, n8, 4)
    g.add_edge(n8, n9, 9)
    g.add_edge(n8, n10, 7)
    g.add_edge(n9, n10, 1)

    # for node in g.get_nodes():
    #     for neighbor in node.get_connections():
    #         print(f'from {node.get_pos()} to {neighbor.get_pos()}: {node.get_weight(neighbor)}')

    curr = breadth_first_search(g, n1, n9)
    while curr != None:
        print(curr.node.pos)
        curr = curr.parent
