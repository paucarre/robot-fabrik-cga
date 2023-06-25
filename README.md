# Introduction
This is a [FABRIK](https://www.researchgate.net/publication/220632147_FABRIK_A_fast_iterative_solver_for_the_Inverse_Kinematics_problem) implementation in [Conformal Geometric Algebra](https://en.wikipedia.org/wiki/Conformal_geometric_algebra), using [clifford library](https://github.com/RobinHankin/clifford), for open chain robots
using [Lie Group](https://vnav.mit.edu/material/04-05-LieGroups-notes.pdf) screw motions from [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) book, using [pytransform3d library](https://dfki-ric.github.io/pytransform3d/).

Most **Inverse Kinematics** (IK) algorithms are based on nonlinear optimization using the the Jacobian. **FABRIK** instead
uses constrained geometry off of the robot's topology to find feasible configurations. This leads to 
fast IK computations and allows extracting additional feasible poses emerging from its symmetries, which
can be used by path planners to constraint motion (e.g. obstacle avoidance). 


**Geometric Algebras** or **Clifford Algebras** are hypercomplex numbers with outer product structure that enables geometric shapes
and operations to be represented algebraically; such as points at infinity, vectors, lines, planes or spheres.
**Conformal Geometric Algebra** (CGA) is a non-degenerate and overdimensioned extension of dual quaternions that 
curves the projected plane into [stereographic projection](https://en.wikipedia.org/wiki/Stereographic_projection) (thus conformal), creating a [null cone](https://en.wikipedia.org/wiki/Light_cone#:~:text=In%20special%20and%20general%20relativity,directions%2C%20would%20take%20through%20spacetime.) analogous
to the one from special relativity with [signature](https://en.wikipedia.org/wiki/Metric_signature) **(4, 1, 0)**. The overdimensioning emerges as the time dimension in special relativity but within engineering and for practical reasons it is often normalized to one, and conceptually detached from time (no time-space fabric is used, only interpreted as time-agnostic geometry). This algebra allows great operability for the 
**FABRIK** algorithm as it has curved shapes, very common in robotics, and operations on them, steaming naturally from geometric algebra. Finally, despite its curved nature, it also allows straight geometries such as lines, vectors or planes to be represented, fundamental as well for robotics and specifically within **FABRIK**
constained geometry.

To manage the [SE(3)](https://www.seas.upenn.edu/~meam620/slides/kinematicsI.pdf) group, the project uses classical Linear Algebra Lie-group theory from [pytransform3d library](https://dfki-ric.github.io/pytransform3d/) as it's a well-tested and robust library. **CGA SE(3)** representation is not used.

For further reading:
- Geometric Algebra
    - [VERSOR Spatial Computing with Conformal Geometric Algebra](http://wolftype.com/versor/colapinto_masters_final_02.pdf)
    - [CGA Chris Doran lecture slides](http://geometry.mrao.cam.ac.uk/wp-content/uploads/2015/11/GA2015_Lecture7.pdf)
    - [Geometric Algebra: An Introduction with Applications in Euclidean and Conformal Geometry](https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=7943&context=etd_theses)
    - [Geometric algebra, conformal geometry and the common curves problem](https://kth.diva-portal.org/smash/get/diva2:1120584/FULLTEXT01.pdf)
    - [A Survey of Geometric Algebra and Geometric Calculus](http://www.faculty.luther.edu/~macdonal/GA&GC.pdf)
- Lie Group Theory
    - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)

# Setup
Currently only **Ubuntu** is supported, but **PR**s to support other Linux distributions
are welcomed but active efforts will be put on dockerizing the solution.

To set up the environment run:
```bash
source ./bin/install-env.sh
```

To run unit tests:
```bash
make test
```
