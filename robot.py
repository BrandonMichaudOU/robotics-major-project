from datetime import datetime
import random
import graph
import numpy as np


# calculates cost to travel along a path
def traverse_path(path, costs):
    cost = 0
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        cost += costs[(start, end)]
    return cost


# runs experiments using specified maps
def run_experiments(topological_sizes, metric_sizes, max_weight_multiplier, dynamic_proportion, num_runs):
    # statistics
    avg_costs = {}
    avg_times = {}

    # run experiment for both topological and metric maps
    for rep in ['t', 'm']:
        sizes = topological_sizes if rep == 't' else metric_sizes  # get sizes of maps

        # run experiment for all sizes of the map
        for size in sizes:
            # store times and costs
            static_times = np.empty((num_runs, 5))
            dynamic_times = np.empty((num_runs, 5))

            static_costs = np.empty((num_runs, 5))
            dynamic_costs = np.empty((num_runs, 5))

            # run experiment specified number times
            for run in range(num_runs):
                # create random map
                g = graph.Graph()
                g.random_topological_map(size, 4 * size, max_weight_multiplier, size, size) if rep == 't' else (
                    g.random_metric_map(size, size, max_weight_multiplier))
                original_costs = g.get_all_connections()

                two_nodes = random.sample(g.get_nodes(), 2)  # pick start and end node

                # plan breadth first search paths
                start_time = datetime.now()
                bfs_path = graph.breadth_first_search(g, two_nodes[0], two_nodes[1])
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][0] = runtime.total_seconds() * 1000
                dynamic_times[run][0] = runtime.total_seconds() * 1000

                # plan depth first search path
                start_time = datetime.now()
                dfs_path = graph.depth_first_search(g, two_nodes[0], two_nodes[1])
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][1] = runtime.total_seconds() * 1000
                dynamic_times[run][1] = runtime.total_seconds() * 1000

                # plan Dijkstra's path
                start_time = datetime.now()
                dijkstra_path = graph.dijkstra(g, two_nodes[0], two_nodes[1])
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][2] = runtime.total_seconds() * 1000
                dynamic_times[run][2] = runtime.total_seconds() * 1000

                # plan A* path
                start_time = datetime.now()
                a_star_path = graph.a_star(g, two_nodes[0], two_nodes[1])
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][3] = runtime.total_seconds() * 1000
                dynamic_times[run][3] = runtime.total_seconds() * 1000

                # traverse breadth first search path
                start_time = datetime.now()
                static_costs[run][0] = traverse_path(bfs_path, original_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][0] += runtime.total_seconds() * 1000

                # traverse depth first search path
                start_time = datetime.now()
                static_costs[run][1] = traverse_path(dfs_path, original_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][1] += runtime.total_seconds() * 1000

                # traverse Dijkstra's path
                start_time = datetime.now()
                static_costs[run][2] = traverse_path(dijkstra_path, original_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][2] += runtime.total_seconds() * 1000

                # traverse A* path
                start_time = datetime.now()
                static_costs[run][3] = traverse_path(a_star_path, original_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][3] += runtime.total_seconds() * 1000

                # traverse D* path
                start_time = datetime.now()
                d_star_path = graph.d_star(g, two_nodes[0], two_nodes[1], original_costs, original_costs)
                static_costs[run][4] = traverse_path(d_star_path, original_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                static_times[run][4] = runtime.total_seconds() * 1000

                # update weight
                g.update_random_weights(dynamic_proportion, max_weight_multiplier)
                updated_costs = g.get_all_connections()

                # traverse breadth first search path
                start_time = datetime.now()
                dynamic_costs[run][0] = traverse_path(bfs_path, updated_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                dynamic_times[run][0] += runtime.total_seconds() * 1000

                # traverse depth first search path
                start_time = datetime.now()
                dynamic_costs[run][1] = traverse_path(dfs_path, updated_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                dynamic_times[run][1] += runtime.total_seconds() * 1000

                # traverse Dijkstra's path
                start_time = datetime.now()
                dynamic_costs[run][2] = traverse_path(dijkstra_path, updated_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                dynamic_times[run][2] += runtime.total_seconds() * 1000

                # traverse A* path
                start_time = datetime.now()
                dynamic_costs[run][3] = traverse_path(a_star_path, updated_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                dynamic_times[run][3] += runtime.total_seconds() * 1000

                # traverse D* path
                start_time = datetime.now()
                d_star_path = graph.d_star(g, two_nodes[0], two_nodes[1], original_costs, updated_costs)
                dynamic_costs[run][4] = traverse_path(d_star_path, updated_costs)
                end_time = datetime.now()
                runtime = end_time - start_time
                dynamic_times[run][4] = runtime.total_seconds() * 1000

            # store average times and costs for experiment in dictionary
            # keys are tuples of the representation, size, and dynamic proportion
            avg_times[(rep, size, 0)] = np.average(static_times, axis=0)
            avg_times[(rep, size, dynamic_proportion)] = np.average(dynamic_times, axis=0)
            avg_costs[(rep, size, 0)] = np.average(static_costs, axis=0)
            avg_costs[(rep, size, dynamic_proportion)] = np.average(dynamic_costs, axis=0)

    return avg_times, avg_costs  # return average times and costs


if __name__ == "__main__":
    # information for map representations
    topological_sizes = [25, 100]
    metric_sizes = [10, 20]
    max_weight_multiplier = 5
    dynamic_proportion = 0.2

    # run experiment using map representation information
    num_runs = 100
    avg_times, avg_costs = run_experiments(topological_sizes, metric_sizes, max_weight_multiplier, dynamic_proportion,
                                           num_runs)

    # print results of experiment
    for rep in ['t', 'm']:
        sizes = topological_sizes if rep == 't' else metric_sizes
        for size in sizes:
            rep_string = 'topological' if rep == 't' else 'metric'
            print(f'For {rep_string} map of size {size} in static environment:')
            print(f'Times: {avg_times[(rep, size, 0)]}')
            print(f'Costs: {avg_costs[(rep, size, 0)]}')
            print()
            print(f'For {rep_string} map of size {size} in dynamic environment:')
            print(f'Times: {avg_times[(rep, size, dynamic_proportion)]}')
            print(f'Costs: {avg_costs[(rep, size, dynamic_proportion)]}')
            print()
