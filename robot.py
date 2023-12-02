

class Robot:
    def __init__(self, graph, start_node, end_node):
        self.graph = graph
        self.current_path = []
        self.current_node = start_node
        self.destination_node = end_node
        self.blocked_list = [] # maybe

    def robot_go(self, algorithm, num_dynamic):
        # Performance metrics
        cost_travelled = 0
        num_iter = 0

        # generate initial path
        self.current_path, iterations = algorithm(self.graph, self.current_node, self.destination_node)
        num_iter += iterations

        # create blocked list (or replace weights in map)
        #   we use a blocked list in addition to editing
        #   the weights directly 
        # TODO: add num_dynamic-many nodes with increased weights

        # Travel through the path
        while current_node != self.destination_node:
            next_node = self.current_path[0]

            # If next node is blocked, recalculate the path and retry
            if next_node in self.blocked_list:
                self.current_path, iterations = algorithm(self.graph, self.current_node, self.destination_node)
                num_iter += iterations
                continue

            # Travel through the node
            cost_travelled += self.graph.get_weight(current_node, next_node)
            current_node = self.current_path.pop(0)

            # Stretch goal: moving blocked nodes
            # TODO