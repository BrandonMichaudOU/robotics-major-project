import math

class MetricMapNode:
    def __init__(self, pos, weight):
        self.pos = pos
        self.weight = weight

class MetricMap:
    def __init__(self, numRows, numCols, defWeight = 1):
        self.map = {}
        self.numRows = numRows
        self.numCols = numCols
        
        for x in range(numRows):
            for y in range(numCols):
                self.map[(x, y)] = MetricMapNode((x, y), defWeight)

    def get_node(self, node_key):
        return self.map[node_key]

    def get_connections(self, node):
        row = node.pos[0]
        col = node.pos[1]
        neighbors = []

        if (row > 0):
            neighbors.append(self.map[(row-1, col)])

        if (col > 0):
            neighbors.append(self.map[(row, col-1)])

        if (row+1 < self.numRows):
            neighbors.append(self.map[(row+1, col)])

        if (col+1 < self.numCols):
            neighbors.append(self.map[(row, col+1)])

        return [neighbor for neighbor in neighbors if neighbor.weight != -1]
            
    def print_weight_map(self):
        for x in range(self.numRows):
            curRow = []
            for y in range(self.numCols):
                curRow.append(self.map[(x, y)])

            print([node.weight for node in curRow])

    def get_nodes(self):
        return list(self.map.values())

    def get_heuristic(self, src_node, dest_node):
        return math.dist(src_node.pos, dest_node.pos)

    def get_weight(self, src_node, dest_node):
        if not dest_node in self.get_connections(src_node):
            return -1
        else:
            return dest_node.weight

    def edit_weight(self, src_node, dest_node, new_weight):
        if not dest_node in self.get_connections(src_node):
            print("Failed to edit connection")
        else:
            dest_node.weight = new_weight

