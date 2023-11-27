import numpy as np

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

    def edit_weight(self, src_node, dest_node, new_weight):
        if not dest_node in self.get_connections(src_node):
            print("Failed to edit connection")
        else:
            src_node.adjacent[dest_node] = new_weight
    
class SearchNode:
    def __init__(self, node, parent):
        self.node = node
        self.parent = parent

# https://en.wikipedia.org/wiki/Breadth-first_search
def breadth_first_search(graph, start, end):
    q = []
    seen = []
    root = SearchNode(start, None)
    q.append(root)
    seen.append(start)
    while len(q) != 0:
        n = q.pop(0)
        if n.node == end:
            break
        for neighbor in graph.get_connections(n.node):
            if neighbor not in seen:
                seen.append(neighbor)
                next = SearchNode(neighbor, n)
                q.append(next)

    # Create a path from start to end
    path = []
    while n != None:
        path.insert(0,n.node)
        n = n.parent

    return path

# https://en.wikipedia.org/wiki/Depth-first_search
def depth_first_search(graph, start, end):
    stack = []
    seen = []
    root = SearchNode(start, None)
    stack.append(root)
    seen.append(start)
    while len(stack) != 0:
        n = stack.pop()
        if n.node == end:
            break
        for neighbor in graph.get_connections(n.node):
            if neighbor not in seen:
                seen.append(neighbor)
                next = SearchNode(neighbor, n)
                stack.append(next)

    # Create a path from start to end
    path = []
    while n != None:
        path.insert(0,n.node)
        n = n.parent

    return path

def dijkstra(graph, start, end):
    print('test')

def euclidean_distance(start, end):
    return ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5

def reconstruct_path(parent, node):
    path = [node]
    while node in parent:
        node = parent[node]
        path.insert(0, node)
    return path

def a_star(graph, start, end):
    open = [start]
    parent = {}
    cost_to_node = {}
    total_cost_estimate = {}

    cost_to_node[start] = 0
    total_cost_estimate[start] = euclidean_distance(start.get_pos(), end.get_pos())

    while len(open) != 0:
        min_node = None
        min_cost = np.inf
        for open_node in open:
            if total_cost_estimate[open_node] < min_cost:
                min_cost = total_cost_estimate[open_node]
                min_node = open_node
        
        if min_node == end:
            return reconstruct_path(parent, min_node)
        
        open.remove(min_node)
        for neighbor in graph.get_connections(min_node):
            temp_cost = cost_to_node[min_node] + min_node.get_weight(neighbor)
            if neighbor not in cost_to_node or temp_cost < cost_to_node[neighbor]:
                parent[neighbor] = min_node
                cost_to_node[neighbor] = temp_cost
                total_cost_estimate[neighbor] = temp_cost + euclidean_distance(neighbor.get_pos(), end.get_pos())
                if neighbor not in open:
                    open.append(neighbor)

    return []

def d_star(graph, start, end):
    print('test')
    
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

    # print("BFS path from end to start")
    # curr = breadth_first_search(g, n1, n9)
    # while curr != None:
    #     print(curr.node.pos)
    #     curr = curr.parent

    # print("\nDFS path from end to start")
    # curr = depth_first_search(g, n1, n9)
    # while curr != None:
    #     print(curr.node.pos)
    #     curr = curr.parent

    for node in a_star(g, n1, n9):
        print(node.get_pos())
