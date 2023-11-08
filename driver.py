import metricmap
import dijkstra
import graph

if __name__ == "__main__":
    # Initialize a 10x10 metric map
    map = metricmap.MetricMap(10, 10)

    point1 = map.get_node((0, 0))
    point2 = map.get_node((0, 2))
    point3 = map.get_node((0, 1))
    
    map.edit_weight(point1, point3, 10)
    print("Metric Map Weights: ")
    map.print_weight_map()
    print()

    # Initialize a topological graph
    g = graph.Graph()

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
    
    # Djikstra's method on a metric map
    path = [node.pos for node in dijkstra.dijkstra(map, point1, point2)]
    path.reverse()
    print(f'From {point1.pos} to {point2.pos}: {path}')

    # Breadth first search on a topological graph
    print("BFS path from end to start")
    curr = graph.breadth_first_search(g, n1, n9)
    while curr != None:
        print(curr.node.pos)
        curr = curr.parent

    # Depth first search on a topological graph
    print("\nDFS path from end to start")
    curr = graph.depth_first_search(g, n1, n9)
    while curr != None:
        print(curr.node.pos)
        curr = curr.parent

    # Breadth first search on a metric map
    print(f'BFS path from {point1.pos} to {point2.pos}:')
    curr = graph.breadth_first_search(map, point1, point2)
    path = []
    while curr != None:
        path.append(curr.node.pos)
        curr = curr.parent
    path.reverse()
    print(path)