class PointChain(object):

    def fromJoints(joint_chain, cga):
        cga = cga
        initial_point = cga.point(0, 0, 0)
        positions = [initial_point]
        transformations = [cga.translator(cga.vector(0.0, 0.0, 0.0))]
        for joint in joint_chain.joints:
            direction = cga.vector(joint.distance, 0.0, 0.0)
            transformations.insert(len(transformations), cga.translator(direction))
            positions.insert(len(positions), cga.sandwiches(initial_point, transformations))
        return PointChain(positions, cga)

    def __init__(self, point_chain, cga):
        self.cga = cga
        self.positions = point_chain

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return f"Points (as vectors): {[self.cga.toVector(point) for point in self.positions]}"

    def last(self):
        return self.positions[len(self.positions) - 1]

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
