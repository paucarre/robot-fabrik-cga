class PointChain(object):

    def __init__(self, joint_chain, cga):
        self.cga = cga
        initial_point = self.cga.point(0, 0, 0)
        self.positions = [initial_point]
        transformations = [self.cga.translation(self.cga.vector(0.0, 0.0, 0.0))]
        for joint in joint_chain.joints:
            direction = self.cga.vector(joint.distance, 0.0, 0.0)
            transformations.insert(len(transformations), self.cga.translation(direction))
            self.positions.insert(len(self.positions), self.cga.sandwiches(initial_point, transformations))

    def __len__(self):
        return len(self.positions)


    def get(self, index, forward):
        if(forward):
            return self.positions[len(self.positions) - 1 - index]
        else:
            return self.positions[index]

    def set(self, index, forward, value):
        if(forward):
            self.positions[len(self.positions) - 1 - index] = value
        else:
            self.positions[index] = value
