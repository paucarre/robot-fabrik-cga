# Introduction

- Lie Group Theory
    - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)
# Major Tasks
 Feature tasks:
 - Convert the main into an actual python object and refactor it as it has been 
 the source of multiple bugs and has the worst code clarity of the codebase
 - Add unit tests in Q-Learning key methods
 - Use Jacobian IK as a deterministic actor network and with it train the critic network
 - Freeze the critic and train an actor network such that it performs better than the 
 deterministic IK Jacobian network
 - Train both the actor and critic network

 Checks, fixes and improvements:
 - Make sure the variance outputed by the actor network is used and the 
 variance is being reduced by entropy maximization.

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
