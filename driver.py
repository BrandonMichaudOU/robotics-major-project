import metricmap
import dijkstra
import graph

if __name__ == "__main__":
    # Initialize a 10x10 metric map
    map = metricmap.MetricMap(3, 3)

    point1 = map.get_node((0, 0))
    point2 = map.get_node((0, 2))
    point3 = map.get_node((0, 1))
    point4 = map.get_node((1, 1))
    
    point3.weight = -1 # Make an obstacle at (0,1)
    point4.weight = 10 # Make difficult terrain at (1,1)
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
    
    # Dijkstra's method on a metric map
    path = [node.pos for node in dijkstra.dijkstra(map, point1, point2)]
    print(f'Metric: Dijkstra From {point1.pos} to {point2.pos}:')
    print(path)

    # Breadth first search on a topological graph
    path = [node.pos for node in graph.breadth_first_search(g, n1, n9)]
    print("Topological: BFS path from end to start")
    print(path)

    # Depth first search on a topological graph
    path = [node.pos for node in graph.depth_first_search(g, n1, n9)]
    print("Topological: DFS path from end to start")
    print(path)

    # Breadth first search on a metric map
    path = [node.pos for node in graph.breadth_first_search(map, point1, point2)]
    print(f'Metric: BFS path from {point1.pos} to {point2.pos}:')
    print(path)

    # Depth first search on a metric map
    path = [node.pos for node in graph.depth_first_search(map, point1, point2)]
    print(f'Metric: DFS path from {point1.pos} to {point2.pos}:')
    print(path)
    