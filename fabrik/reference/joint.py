class Joint(object):
    def __init__(self, angle_constraint, distance):
        self.angle_constraint = angle_constraint
        self.distance = distance

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return (
            f"[ Distance: {self.distance}, Angle Constraint: {self.angle_constraint} ]"
        )
