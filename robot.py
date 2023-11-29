

class Robot:
    def __init__(self):
        self.map = None
        self.current_path = []
        self.current_node = None
        self.destination_node = None
        self.blocked_list = [] # optional

    def robot_go(self):
        # generate map (maybe in constructor?)

        # place robot in map and choose destination (maybe in constructor?)

        # generate initial path

        # create blocked list (or replace weights in map)

        # loop to go through path
            # check if next node is a blocked node
                # if yes, recalculate route