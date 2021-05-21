

## Environment

Since CE-Q and Foe-Q use linear programming to find optimal polcies, `cvxpy` is required for this project. The program uses `GLPK` solver which can be installed with package `cvxopt`. To install the required packages, run

```shell
pip install cvxpy
pip install cvxopt
pip install numpy
```

If the `GLPK` solver is not working, refer to [this link](https://www.cvxpy.org/install/index.html#install-with-cvxopt-and-glpk-support) for more information and help.

## Execution

To train the agents, open the jupyter notebook `soccer.ipynb` which contains interactive interface to adjust the parameters. Follow the steps to replicate the figures. 