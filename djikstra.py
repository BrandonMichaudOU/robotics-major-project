import sys

# https://www.programiz.com/dsa/dijkstra-algorithm
def djikstra(map, start_node, end_node):
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

        print(f'min node: {min_node.pos}')
        # add node's neighbors and update distances
        for neighbor in map.get_connections(min_node):
            temp_dist = distances[min_node] + map.get_weight(min_node, neighbor)
            print(f'Distances[{min_node.pos}] = {distances[min_node]}; link: {map.get_weight(min_node, neighbor)}')
            print(f'Temp distance: {temp_dist}')
            print(f'Old distance: {distances[neighbor]}')
            if temp_dist < distances[neighbor]:
                print(f'updating node: {neighbor.pos}')
                distances[neighbor] = temp_dist
                parents[neighbor] = min_node

        # remove node from priority list
        pqueue.remove(min_node)

    # Make list of nodes from end_node to start
    cur_node = end_node
    shortest_path = [cur_node]
    while cur_node != start_node or parents[cur_node] != None:
        shortest_path.append(parents[cur_node])
        cur_node = parents[cur_node]
    return shortest_path
