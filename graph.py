import numpy as np
import random
import copy


class Node:
    def __init__(self, pos):
        self.pos = pos  # position of node
        self.adjacent = {}  # dictionary from node to weight

    def get_pos(self):
        return self.pos

    def get_connections(self):
        return self.adjacent.keys()  

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def add_neighbor(self, neighbor, weight=1):
        self.adjacent[neighbor] = weight


class Graph:
    def __init__(self):
        self.nodes = []  # list of nodes in graph
        self.num_nodes = 0
        self.num_edges = 0
        self.x_pos_max = 0
        self.y_pos_max = 0
        self.topological = False  # flag indicating the map is topological
        self.metric = False  # flag indicating the map is metric

    def get_nodes(self):
        return self.nodes

    def get_weight(self, src_node, dest_node):
        return src_node.adjacent[dest_node]

    def get_connections(self, source):
        return source.adjacent.keys()

    def get_all_connections(self):
        all_connections = {}
        for node in self.nodes:
            connections = self.get_connections(node)
            for connection in connections:
                all_connections[(node, connection)] = node.get_weight(connection)
        return all_connections  # dictionary from two nodes to weight

    def add_node(self, pos):
        self.num_nodes = self.num_nodes + 1
        new_node = Node(pos)
        self.nodes.append(new_node)
        return new_node

    def add_edge(self, source, dest, weight=1):
        # if the nodes are not in the graph add them
        if source not in self.nodes:
            self.add_node(source)
        if dest not in self.nodes:
            self.add_node(dest)

        # increment number of edges if edges did not exist
        if source not in dest.get_connections():
            self.num_edges += 1

        # add edge in both directions
        source.add_neighbor(dest, weight)
        dest.add_neighbor(source, weight)

    # reset graph
    def clear(self):
        self.nodes.clear()
        self.num_nodes = 0
        self.num_edges = 0
        self.x_pos_max = 0
        self.y_pos_max = 0
        self.topological = False
        self.metric = False

    def random_topological_map(self, num_nodes, num_edges, max_weight_multiplier, x_pos_max, y_pos_max):
        # if the number of edges is not possible, return
        if num_edges < num_nodes - 1 or num_edges > num_nodes * (num_nodes - 1) / 2:
            return None

        # clear graph and indicate it is topological
        self.clear()
        self.topological = True
        self.metric = False
        self.x_pos_max = x_pos_max
        self.y_pos_max = y_pos_max

        previous = None  # keeps track of previous node
        positions = []  # keeps track of all used node positions
        connections = []  # keeps track of connected nodes

        # set weight bounds based on maximum distance
        max_distance = euclidean_distance((0, 0), (x_pos_max, y_pos_max))
        min_weight = int(max_distance) + 1
        max_weight = int(random.uniform(1, max_weight_multiplier) * min_weight)

        # randomly generate each node and connect it to previous
        for i in range(num_nodes):
            # generate new random position for node
            pos = (random.randrange(0, x_pos_max), random.randrange(0, y_pos_max))
            while pos in positions:
                pos = (random.randrange(0, x_pos_max), random.randrange(0, y_pos_max))

            # add node
            positions.append(pos)
            n = self.add_node(pos)

            # connect node to previous node to ensure fully-connected graph
            if previous is not None:
                weight = random.randrange(min_weight, max_weight)  # generate random weight

                # add edge
                connections.append((i - 1, i))
                connections.append((i, i - 1))
                self.add_edge(n, previous, weight)
                num_edges -= 1

            previous = n

        # randomly generate remaining edges
        for _ in range(num_edges):
            # randomly select two new different nodes
            i = random.randrange(0, num_nodes - 1)
            j = random.randrange(0, num_nodes - 1)
            while i == j or (i, j) in connections:
                i = random.randrange(0, num_nodes - 1)
                j = random.randrange(0, num_nodes - 1)

            weight = random.randrange(min_weight, max_weight)  # generate random weight

            # add edge
            connections.append((i, j))
            connections.append((j, i))
            self.add_edge(self.nodes[i], self.nodes[j], weight)

    def random_metric_map(self, num_rows, num_cols, max_weight_multiplier):
        # clear graph and indicate it is a metric map
        self.clear()
        self.topological = False
        self.metric = True

        # set weight bounds based on maximum distance
        min_weight = 1
        max_weight = int(random.uniform(1, max_weight_multiplier) * min_weight)

        # create 2d grid
        grid = [[Node((0, 0)) for a in range(num_rows)] for b in range(num_cols)]

        # add nodes for each cell in grid
        for i in range(num_rows):
            for j in range(num_cols):
                pos = (i, j)
                n = self.add_node(pos)
                grid[i][j] = n

        # add weights for each adjacent cell
        for i in range(num_rows):
            for j in range(num_cols):
                weight = random.randrange(min_weight, max_weight)  # generate random weight
                if i > 0:
                    grid[i - 1][j].add_neighbor(grid[i][j], weight)
                if j > 0:
                    grid[i][j - 1].add_neighbor(grid[i][j], weight)
                if i < num_rows - 1:
                    grid[i + 1][j].add_neighbor(grid[i][j], weight)
                if j < num_cols - 1:
                    grid[i][j + 1].add_neighbor(grid[i][j], weight)

    def update_random_weights(self, proportion, max_weight_multiplier):
        if self.metric:
            num_updates = round(proportion * self.num_nodes)  # random number of nodes to update

            # set weight bounds based on maximum distance
            min_weight = 1
            max_weight = int(random.uniform(1, max_weight_multiplier) * min_weight)

            # randomly update node weights
            for _ in range(num_updates):
                node = random.choice(self.nodes)  # choose random node to update
                weight = random.randrange(min_weight, max_weight)  # generate random weight

                # update weight of node to all adjacent nodes
                for neighbor in node.get_connections():
                    neighbor.add_neighbor(node, weight)
        else:
            num_updates = round(proportion * self.num_edges)  # random number of weights to update

            # set weight bounds based on maximum distance
            max_distance = euclidean_distance((0, 0), (self.x_pos_max, self.y_pos_max))
            min_weight = int(max_distance) + 1
            max_weight = int(random.uniform(1, max_weight_multiplier) * min_weight)

            # randomly update edge weights
            for _ in range(num_updates):
                # select random pair of connected nodes
                node1 = random.choice(self.nodes)
                node2 = random.choice(list(node1.get_connections()))

                weight = random.randrange(min_weight, max_weight)  # generate random weight

                self.add_edge(node1, node2, weight)  # update weight of edge


# given the end node and the parents, reconstructs path from start to end
def reconstruct_path(parent, node):
    path = [node]
    while node in parent:
        node = parent[node]
        path.insert(0, node)
    return path


# https://en.wikipedia.org/wiki/Breadth-first_search
def breadth_first_search(graph, start, end):
    q = [start]  # queue of nodes
    seen = [start]  # list of nodes seen
    parent = {}  # dictionary from node to parent node

    # statistics
    iterations = 0

    # add unseen nodes until goal is found or queue is empty
    while q:
        iterations += 1

        n = q.pop(0)  # get node at front of queue

        # if the node is the goal, return path and iterations
        if n == end:
            return reconstruct_path(parent, n), iterations

        # add all unseen neighbors of node to back of queue
        for neighbor in graph.get_connections(n):
            if neighbor not in seen:
                seen.append(neighbor)
                parent[neighbor] = n
                q.append(neighbor)

    return [], iterations  # return empty path and iterations if path not found


# https://en.wikipedia.org/wiki/Depth-first_search
def depth_first_search(graph, start, end):
    stack = [start]  # stack of nodes
    seen = [start]  # list of nodes seen
    parent = {}  # dictionary from node to parent node

    # statistics
    iterations = 0

    # add unseen nodes until goal is found or stack is empty
    while stack:
        iterations += 1

        n = stack.pop()  # get node from top of stack

        # if the node is the goal, return path and iterations
        if n == end:
            return reconstruct_path(parent, n), iterations

        # add all unseen neighbors of node to top of stack
        for neighbor in graph.get_connections(n):
            if neighbor not in seen:
                seen.append(neighbor)
                parent[neighbor] = n
                stack.append(neighbor)

    return [], iterations  # return empty path and iterations if path not found


# https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
def dijkstra(graph, start, end):
    distances = {}  # dictionary from node to distance
    parent = {}  # dictionary from node to parent node
    pqueue = []  # priority queue of nodes using distance as key

    # add all nodes to priority queue and indicate they cannot be reach
    for node in graph.get_nodes():
        distances[node] = np.inf
        pqueue.append(node)

    distances[start] = 0  # set distance to start to be 0

    # statistics
    iterations = 0

    # update distances until priority queue is empty or goal reached
    while pqueue:
        iterations += 1

        # find the node in the priority queue with minimum distance
        min_node = None
        for node in pqueue:
            if min_node is None:
                min_node = node
            elif distances[node] < distances[min_node]:
                min_node = node

        # if the minimum distance node is the goal, return path and iterations
        if min_node == end:
            return reconstruct_path(parent, end), iterations

        # find total distances to neighbor of minimum distance node and update them if shorter
        for neighbor in graph.get_connections(min_node):
            temp_dist = distances[min_node] + graph.get_weight(min_node, neighbor)
            if temp_dist < distances[neighbor]:
                distances[neighbor] = temp_dist
                parent[neighbor] = min_node

        pqueue.remove(min_node)  # remove minimum distance node from priority queue

    return [], iterations  # return empty path and iterations if path not found


# finds straight line distance between two points
def euclidean_distance(start, end):
    return ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5


# https://en.wikipedia.org/wiki/A*_search_algorithm
# uses Euclidean distance as heuristic
def a_star(graph, start, end):
    open = [start]  # priority queue of open nodes with total cost estimate as key
    parent = {}  # dictionary from node to parent node
    cost_to_node = {start: 0}  # dictionary from node to cost from start to node

    # dictionary from node to total cost estimate
    total_cost_estimate = {start: euclidean_distance(start.get_pos(), end.get_pos())}

    # statistics
    iterations = 0

    # update distances until priority queue is empty or goal reached
    while open:
        iterations += 1

        # find the node in the priority queue with minimum total cost estimate
        min_node = None
        min_cost = np.inf
        for open_node in open:
            if total_cost_estimate[open_node] < min_cost:
                min_cost = total_cost_estimate[open_node]
                min_node = open_node

        # if the minimum total cost estimate node is the goal, return path and iterations
        if min_node == end:
            return reconstruct_path(parent, min_node), iterations

        open.remove(min_node)  # remove minimum total cost estimate node from priority queue

        # find total distances to neighbor of minimum total cost estimate node and update them if shorter
        for neighbor in graph.get_connections(min_node):
            temp_cost = cost_to_node[min_node] + min_node.get_weight(neighbor)
            if neighbor not in cost_to_node or temp_cost < cost_to_node[neighbor]:
                parent[neighbor] = min_node
                cost_to_node[neighbor] = temp_cost
                total_cost_estimate[neighbor] = temp_cost + euclidean_distance(neighbor.get_pos(), end.get_pos())
                if neighbor not in open:
                    open.append(neighbor)

    return [], iterations  # return empty path and iterations if path not found


# https://www.ri.cmu.edu/pub_files/pub3/stentz_anthony__tony__1994_2/stentz_anthony__tony__1994_2.pdf
def d_star(graph, start, end):
    open = []  # list of open nodes
    closed = []  # list of closed nodes
    new = []  # list of new nodes

    c = graph.get_all_connections()  # original edges in graph
    b = {}  # back pointers to next node
    k = {end: 0}  # lowest cost estimate from node to goal
    h = {}  # heuristic estimate of cost from node to goal

    # update heuristic estimate for each node and indicate it has no parent
    for n in graph.get_nodes():
        b[n] = None
        h[n] = euclidean_distance(n.get_pos(), end.get_pos())

    # find the node with minimum cost estimate from open list and return node with cost
    def min_state_val():
        if not open:
            return None, -1
        min_node = None
        min_cost = np.inf
        for open_node in open:
            if k[open_node] < min_cost:
                min_cost = k[open_node]
                min_node = open_node
        return min_node, min_cost

    # find the node with minimum cost estimate from open list and return node
    def min_state():
        state, _ = min_state_val()
        return state

    # find the node with minimum cost estimate from open list and return cost
    def min_val():
        _, value = min_state_val()
        return value

    # removes node from open list and adds it to closed
    def delete(x):
        open.remove(x)
        closed.append(x)

    # insert node into open list with new estimated cost
    def insert(x, h_new):
        if x in new:  # if node is in new list
            k[x] = h_new  # update lowest cost estimate

            # move node from new list to open list
            new.remove(x)
            open.append(x)
        elif x in open:  # if node is in open
            k[x] = min(k[x], h_new)  # update lowest cost estimate
        else:  # if node is in closed
            k[x] = min(h[x], h_new)  # update lowest cost estimate

            # move node from closed list to open list
            closed.remove(x)
            open.append(x)

        h[x] = h_new  # update estimate cost of node

    # proces minimum cost estimate node from open list
    def process_state():
        x = min_state()  # get minimum cost estimate node from open list

        # if there is no minimum cost estimate node from open list indicate so
        if x is None:
            return -1

        k_old = k[x]  # store minium cost estimate of minimum cost estimate node
        delete(x)  # move minimum cost estimate node from open list to closed list

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

    # update cost between two nodes and propagate changes
    def modify_cost(x, y, cval):
        c[(x, y)] = cval
        if x in closed:
            insert(x, h[x])
        return min_val()

    # tests if a < b
    def less(a, b):
        if a < b:
            return True
        return False

    # simulates the path traversal
    def move_robot():
        # add all nodes to new list
        for n in graph.get_nodes():
            new.append(n)

        insert(end, 0)  # move end node to open list
        val = 0  # keep track of process state return values

        # while path is not found and can be found, process the state
        while start not in closed and val != -1:
            val = process_state()

        # if no path was found, return empty path
        if start in new:
            return []

        r = start  # current node in simulation
        path = [r]  # path found
        s = c  # TODO: updated edges after environment changed

        # construct path from start to goal
        while r != end:
            # if the weight of an edge has changed, update original edges
            for connection in c.keys():
                if s[connection] != c[connection]:
                    val = modify_cost(connection[0], connection[1], s[connection])

            # propagate changes of updated edge
            while less(val, h[r]) and val != -1:
                val = process_state()

            r = b[r]  # move to next node
            path.append(r)  # add node to path

        return path  # return path from start to goal

    return move_robot()  # return path from start to goal


if __name__ == '__main__':
    g = Graph()
    g.random_topological_map(1000, 3000, 100, 4000, 4000)
    # g.random_metric_map(4, 4, 10)

    # n1 = g.add_node((0, 0))
    # n2 = g.add_node((5, 2))
    # n3 = g.add_node((5, -2))
    # n4 = g.add_node((8, 0))
    # n5 = g.add_node((10, 1))
    # n6 = g.add_node((12, 2))
    # n7 = g.add_node((12, 0))
    # n8 = g.add_node((15, 0))
    # n9 = g.add_node((20, 0))
    # n10 = g.add_node((20, -2))
    #
    # g.add_edge(n1, n2, 2)
    # g.add_edge(n1, n3, 5)
    # g.add_edge(n2, n3, 1)
    # g.add_edge(n2, n4, 3)
    # g.add_edge(n2, n5, 1)
    # g.add_edge(n3, n4, 2)
    # g.add_edge(n3, n7, 6)
    # g.add_edge(n3, n10, 30)
    # g.add_edge(n4, n5, 1)
    # g.add_edge(n4, n7, 8)
    # g.add_edge(n5, n6, 7)
    # g.add_edge(n6, n7, 1)
    # g.add_edge(n6, n8, 6)
    # g.add_edge(n7, n8, 4)
    # g.add_edge(n8, n9, 9)
    # g.add_edge(n8, n10, 7)
    # g.add_edge(n9, n10, 1)

    # print('Before')
    # before_connections = g.get_all_connections()
    # for node1, node2 in before_connections:
    #     print(f'from {node1.get_pos()} to {node2.get_pos()}: {before_connections[(node1, node2)]}')
    #
    # g.update_random_weights(0.7, 20)
    #
    # print('\nAfter')
    # after_connections = g.get_all_connections()
    # for node1, node2 in after_connections:
    #     print(f'from {node1.get_pos()} to {node2.get_pos()}: {before_connections[(node1, node2)]}')

    n1 = random.choice(g.get_nodes())
    n9 = random.choice(g.get_nodes())
    print('BFS')
    nodes, _ = breadth_first_search(g, n1, n9)
    for node in nodes:
        print(node.get_pos())
    print()

    print('DFS')
    nodes, _ = depth_first_search(g, n1, n9)
    for node in nodes:
        print(node.get_pos())
    print()

    print('Dijkstra')
    nodes, _ = dijkstra(g, n1, n9)
    for node in nodes:
        print(node.get_pos())
    print()

    print('A*')
    nodes, _ = a_star(g, n1, n9)
    for node in nodes:
        print(node.get_pos())
    print()

    print('D*')
    for node in d_star(g, n1, n9):
        print(node.get_pos())
