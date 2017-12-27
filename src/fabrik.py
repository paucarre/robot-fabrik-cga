from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from JointChain import JointChain, Joint
from FabrikSolver import FabrikSolver

if __name__ == "__main__":
    cga = ConformalGeometricAlgebra()
    first_joint = Joint(2.0, 100.0)
    second_joint = Joint(2.0, 100.0)
    joint_chain = JointChain([first_joint, second_joint])
    target_point = cga.point(70.0, 20.0, 0.0)
    fabrik_solver = FabrikSolver()
    positions = fabrik_solver.solve(joint_chain, target_point)
    print(positions)
