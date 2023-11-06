import metricmap
import djikstra

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
