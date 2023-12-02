from datetime import datetime
import random
import graph

class Robot:
    def __init__(self, g, start_node, end_node):
        self.g = g
        self.current_path = []
        self.current_node = start_node
        self.destination_node = end_node
        self.blocked_list = [] # maybe

    def robot_go(self, algorithm, proportion_dynamic, max_weight):
        # Performance metrics
        cost_travelled = 0
        start_time = datetime.now()

        # Generate initial path
        self.current_path = algorithm(self.g, self.current_node, self.destination_node)

        # Randomly add 'dynamic' obstacles to the map
        self.g.update_random_weights(proportion_dynamic, max_weight)

        # Travel through the path
        self.current_path.pop(0) # remove starting node
        while self.current_node != self.destination_node:
            next_node = self.current_path[0]
            
            # Travel through the node
            cost_travelled += self.g.get_weight(self.current_node, next_node)
            self.current_node = self.current_path.pop(0)

        end_time = datetime.now()
        runtime = end_time - start_time
        return cost_travelled, runtime.total_seconds() *1000


def run_experiments(algorithm, num_runs = 100):
    avg_costs = []
    avg_runtimes = []
    

    # Topological, small, static
    size = 25
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_topological_map(size, size*2, 2, 10, 10)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, 0, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Topological, small, dynamic
    size = 25
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_topological_map(size, size*2, 2, 10, 10)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, .10, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Topological, big, static
    size = 100
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_topological_map(size, size*2, 2, 10, 10)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, 0, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Topological, big, dynamic
    size = 100
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_topological_map(size, size*2, 2, 10, 10)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, .10, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)


    # Metric, small, static
    num_cols = 10
    num_rows = 10
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_metric_map(num_rows, num_cols, 2)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, 0, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Metric, small, dynamic
    num_cols = 10
    num_rows = 10
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_metric_map(num_rows, num_cols, 2)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, .10, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Metric, big, static
    num_cols = 20
    num_rows = 20
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_metric_map(num_rows, num_cols, 2)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, 0, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    # Metric, big, dynamic
    num_cols = 20
    num_rows = 20
    costs = []
    runtimes = []

    
    for _ in range(num_runs):
        g = graph.Graph()
        g.random_metric_map(num_rows, num_cols, 10)
        two_nodes = random.sample(g.get_nodes(), 2)
        robot = Robot(g, two_nodes[0], two_nodes[1])
        c, r = robot.robot_go(algorithm, .10, 100)
        costs.append(c)
        runtimes.append(r)
    avg_costs.append(sum(costs)/num_runs)
    avg_runtimes.append(sum(runtimes)/num_runs)

    return avg_costs, avg_runtimes


if __name__ == "__main__":
    print("Breadth-first search experiments")
    c, r = run_experiments(graph.breadth_first_search, 200)
    print(c)
    print(r)