import metricmap
import djikstra
import graph

if __name__ == "__main__":
    map = metricmap.MetricMap(10, 10)
    print("Metric Map Weights: ")
    map.print_weight_map()
    print()

    point1 = map.get_node((0, 0))
    point2 = map.get_node((3, 4))
    point3 = map.get_node((1, 0))

    print(f'P1 {point1.pos} to P2 {point2.pos}')
    print(f'Distance: {map.get_heuristic(point1, point2)}; Weight: {map.get_weight(point1, point2)}')
    print()
    print(f'P1 {point1.pos} to P3 {point3.pos}')
    print(f'Distance: {map.get_heuristic(point1, point3)}; Weight: {map.get_weight(point1, point3)}')

    map.edit_weight(point1, point3, 10)
    print(f'P1 {point1.pos} to P3 {point3.pos}')
    print(f'Distance: {map.get_heuristic(point1, point3)}; Weight: {map.get_weight(point1, point3)}')

    path = [node.pos for node in djikstra.djikstra(map, point1, point2)]
    path.reverse()
    print(path)

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

    # for node in g.get_nodes():
    #     for neighbor in node.get_connections():
    #         print(f'from {node.get_pos()} to {neighbor.get_pos()}: {node.get_weight(neighbor)}')

    print("BFS path from end to start")
    curr = graph.breadth_first_search(g, n1, n9)
    while curr != None:
        print(curr.node.pos)
        curr = curr.parent

    print("\nDFS path from end to start")
    curr = graph.depth_first_search(g, n1, n9)
    while curr != None:
        print(curr.node.pos)
        curr = curr.parent