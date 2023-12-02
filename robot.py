from datetime import datetime

class Robot:
    def __init__(self, graph, start_node, end_node):
        self.graph = graph
        self.current_path = []
        self.current_node = start_node
        self.destination_node = end_node
        self.blocked_list = [] # maybe

    def robot_go(self, algorithm, proportion_dynamic, max_weight):
        # Performance metrics
        cost_travelled = 0
        num_iter = 0
        start_time = datetime.now()

        # Generate initial path
        self.current_path, iterations = algorithm(self.graph, self.current_node, self.destination_node)
        num_iter += iterations

        # Randomly add 'dynamic' obstacles to the map
        self.graph.update_random_weight(proportion_dynamic, max_weight)

        # Travel through the path
        while self.current_node != self.destination_node:
            next_node = self.current_path[0]
            
            # Travel through the node
            cost_travelled += self.graph.get_weight(current_node, next_node)
            current_node = self.current_path.pop(0)

        end_time = datetime.now()
        runtime = end_time - start_time
        return cost_travelled, num_iter, runtime