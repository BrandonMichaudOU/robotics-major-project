import sys

# https://www.programiz.com/dsa/dijkstra-algorithm
def dijkstra(map, start_node, end_node):
    nodes_seen = []
    distances = {}
    parents = {}
    pqueue = []

    for node in map.get_nodes():
        distances[node] = sys.maxsize
        pqueue.append(node)
    distances[start_node] = 0
    parents[start_node] = None

    while pqueue:
        # get the 'smallest' node
        min_node = None
        for node in pqueue:
            if min_node == None:
                min_node = node
            elif distances[node] < distances[min_node]:
                min_node = node
                
        # add node's neighbors and update distances
        for neighbor in map.get_connections(min_node):
            temp_dist = distances[min_node] + map.get_weight(min_node, neighbor)
            if temp_dist < distances[neighbor]:
                distances[neighbor] = temp_dist
                parents[neighbor] = min_node

        # remove node from priority list
        nodes_seen.append(min_node) # add to nodes seen
        pqueue.remove(min_node)

    # Make list of nodes from start to end
    cur_node = end_node
    shortest_path = [cur_node]
    while cur_node != start_node or parents[cur_node] != None:
        shortest_path.insert(0,parents[cur_node])
        cur_node = parents[cur_node]
    return shortest_path, nodes_seen
