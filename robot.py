

class Robot:
    def __init__(self, map, start_node, end_node):
        self.map = map
        self.current_path = []
        self.current_node = start_node
        self.destination_node = end_node
        self.blocked_list = [] # maybe

    def robot_go(self, algorithm, num_dynamic):
        # Performance metrics
        cost_travelled = 0
        nodes_seen = []

        # generate initial path
        self.current_path, seen = algorithm(self.map, self.current_node, self.destination_node)
        for node in seen:
            if node not in nodes_seen:
                nodes_seen.append(node)

        # create blocked list (or replace weights in map)
        # TODO: add num_dynamic-many nodes with increased weights

        while current_node != self.destination_node:
            next_node = self.current_path[0]
            if next_node in self.blocked_list:
                self.current_path, seen = algorithm(self.map, self.current_node, self.destination_node)
                for node in seen:
                    if node not in nodes_seen:
                        nodes_seen.append(node)
                continue
            cost_travelled += 
            current_node = self.current_path.pop(0)

        # loop to go through path
            # check if next node is a blocked node
                # if yes, recalculate route