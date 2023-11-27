import numpy as np


class Node:
    def __init__(self, pos):
        self.pos = pos
        self.adjacent = {}  # dictionary from node to weight

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
        self.nodes = []  # List of nodes in graph
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

    def get_all_connections(self):
        all_connections = {}
        for node in self.nodes:
            connections = self.get_connections(node)
            for connection in connections:
                all_connections[(node, connection)] = node.get_weight(connection)
        return all_connections

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
            return n
        for neighbor in graph.get_connections(n.node):
            if neighbor not in seen:
                seen.append(neighbor)
                next_node = SearchNode(neighbor, n)
                q.append(next_node)


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
            return n
        for neighbor in graph.get_connections(n.node):
            if neighbor not in seen:
                seen.append(neighbor)
                next_node = SearchNode(neighbor, n)
                stack.append(next_node)


def dijkstra(graph, start, end):
    print('test')


def euclidean_distance(start, end):
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5


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
    total_cost_estimate = {start: euclidean_distance(start.get_pos(), end.get_pos())}

    cost_to_node[start] = 0

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
    open = []
    closed = []
    new = []

    c = graph.get_all_connections()
    b = {}  # backpointers to next node
    k = {end: 0}  # cost from node to goal
    h = {}  # heuristic estimate of cost from node to goal
    for n in graph.get_nodes():
        b[n] = None
        h[n] = euclidean_distance(n.get_pos(), end.get_pos())

    def min_state_val():
        if len(open) == 0:
            return None, -1
        min_node = None
        min_cost = np.inf
        for open_node in open:
            if k[open_node] < min_cost:
                min_cost = k[open_node]
                min_node = open_node
        return min_node, min_cost

    def min_state():
        state, _ = min_state_val()
        return state

    def min_val():
        _, value = min_state_val()
        return value

    def delete(x):
        open.remove(x)
        closed.append(x)

    def insert(x, h_new):
        if x in new:
            k[x] = h_new
            new.remove(x)
            open.append(x)
        elif x in open:
            k[x] = min(k[x], h_new)
        else:
            k[x] = min(h[x], h_new)
            closed.remove(x)
            open.append(x)
        h[x] = h_new

    def process_state():
        x = min_state()

        if x is None:
            return -1

        k_old = k[x]
        delete(x)

        if k_old < h[x]:
            for y in graph.get_connections(x):
                if y not in new and h[y] <= k_old and h[x] > h[y] + c[(x, y)]:
                    b[x] = y
                    h[x] = h[y] + c[(x, y)]
        if k_old == h[x]:
            for y in graph.get_connections(x):
                if (y in new or (b[y] == x and h[y] != h[x] + c[(x, y)]) or
                        (b[y] != x and h[y] > h[x] + c[(x, y)])):
                    b[y] = x
                    insert(y, h[x] + c[(x, y)])
        else:
            for y in graph.get_connections(x):
                if y in new or (b[y] == x and h[y] != h[x] + c[(x, y)]):
                    b[y] = x
                    insert(y, h[x] + c[(x, y)])
                else:
                    if b[y] != x and h[y] > h[x] + c[(x, y)] and x in closed:
                        insert(x, h[x])
                    else:
                        if b[y] != x and h[x] > h[y] + c[(x, y)] and y in closed and h[y] > k_old:
                            insert(y, h[y])
        return min_val()

    def modify_cost(x, y, cval):
        c[(x, y)] = cval
        if x in closed:
            insert(x, h[x])
        return min_val()

    def less(a, b):
        if a < b:
            return True
        return False

    def move_robot():
        for n in graph.get_nodes():
            new.append(n)
        insert(end, 0)
        val = 0
        while start not in closed and val != -1:
            val = process_state()
        if start in new:
            return None
        r = start
        path = [r]
        s = c
        while r != end:
            for connection in c.keys():
                if s[connection] != c[connection]:
                    val = modify_cost(connection[0], connection[1], s[connection])
            while less(val, h[r]) and val != -1:
                val = process_state()
            r = b[r]
            path.append(r)
        return path

    return move_robot()


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

    print('A*')
    for node in a_star(g, n1, n9):
        print(node.get_pos())

    print('D*')
    for node in d_star(g, n1, n9):
        print(node.get_pos())