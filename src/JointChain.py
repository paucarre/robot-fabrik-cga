class Joint(object):

    def __init__(self, angle_constraint, distance):
        self.angle_constraint = angle_constraint
        self.distance = distance

    def jointAt(self, index, forward):
        if(forward):
            return self.joints[len(self.joints) - 1 - index]
        else:
            return self.joints[index]

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return f"[ Distance: {self.distance}, Angle Constraint: {self.angle_constraint} ]"

#TODO: Currently only supports planar robots. Need to support full D-H topology
class JointChain(object):

    def __init__(self, joints):
        self.joints = joints

    def __len__(self):
        return len(self.joints)

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        joints_as_stirng = [f"{joint}" for joint in self.joints]
        return f"Joints: {joints_as_stirng}"

    def last(self):
        return self.joints[len(self.joints) - 1]

    def get(self, index, forward):
        if(forward):
            return self.joints[len(self.joints) - 1 - index]
        else:
            return self.joints[index]
