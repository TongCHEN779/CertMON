# Semialgebraic Representation of Monotone Deep Equilibrium Models and Applications to Certification
This repository is the code for submission 4075 at NeurIPS 2021.

## Description of files
*params_ellips* is a folder containing all the parameters of ellipsoids in different cases;

*Ellipsoid.m* (in Matlab) contains the code for Ellipsoid Model;

*solution_fix_point.m* (in Matlab) contains the code for fixed-point iteration;

*MonDEQ.jl* (in Julia) contains the code for Certification and Lipschitz Models;

*deq_MNIST_SingleFcNet_m=20.mat* contains the parameters of a pre-trained single fully-connected monDEQ;

*MNIST_\*.mat* contains the training and testing MNIST dataset.

*attack_\*.mat* contains several adversarial examples for the first test MNIST example;

*border_\*.mat* contains samples of the image of the input region.

## Dependencies
For Julia, it requires to install the following packages: *JuMP, LinearAlgebra, MosekTools (need license), MAT, MLDatasets*;

For Matlab, it requires to install the [CVX](http://cvxr.com/cvx/) package and [Mosek](https://www.mosek.com/) solver (need license).

## Certification Model
The following code allows us to solve problem (CertMON-i) for the k-th test example in the paper:

```Julia
test_x, test_y = MNIST.testdata();
vars = matread("deq_MNIST_SingleFcNet_m=20.mat");
A = vars["A"]; B = vars["B"]; U = vars["U"]; u = vars["u"]; C = vars["C"];
p = size(U, 1); m = 20; W = (1-m)*Matrix(I(p))-A'*A+B-B';
me = 0.1307; std = 0.3081; e = 0.1/std;
k = 1; x0 = (vec(test_x[:,:,k]).-me)./std; i = 1;
obj = cert_monDEQ(x0, W, U, u, C[i,:], e; nrm = "linf")
```

## Lipschitz Model
The following code allows us to compute the upper bound of global Lipschitz constant of monDEQ:

```Julia
vars = matread("deq_MNIST_SingleFcNet_m=20.mat")
A = vars["A"]; B = vars["B"]; U = vars["U"]; u = vars["u"]; C = vars["C"];
p = size(U, 1); m = 20; W = (1-m)*Matrix(I(p))-A'*A+B-B';
std = 0.3081; e = 0.1/std;
obj = lip_monDEQ(W, U, u, C, e; nrm = "linf")
```

## Ellipsoid Model
Run directly the code in the first section of *Ellipsoid.m*.

## Certifying Robustness
For Certification Model, use the following code to compute the ratio of the (first 100) test examples:

```Julia
test_x, test_y = MNIST.testdata();
vars = matread("deq_MNIST_SingleFcNet_m=20.mat")
A = vars["A"]; B = vars["B"]; U = vars["U"]; u = vars["u"]; C = vars["C"];
p = size(U, 1); m = 20; W = (1-m)*Matrix(I(p))-A'*A+B-B';
n = 100; std = 0.3081; e = 0.1/std;
r = cert(W, U, u, C, n, e, test_x, test_y)
```

For Lipschitz Model, use the following code to compute the ratio of the (first 100) test examples:

```Julia
test_x, test_y = MNIST.testdata()
vars = matread("deq_MNIST_SingleFcNet_m=20.mat")
A = vars["A"]; B = vars["B"]; U = vars["U"]; u = vars["u"]; C = vars["C"]; c = vars["c"];
p = size(U, 1); m = 20; W = (1-m)*Matrix(I(p))-A'*A+B-B';
n = 100; std = 0.3081; e = 0.1/std; L = 4.67;
r = cert_lip(W, U, u, C, c, n, e, L, test_x, test_y)
```

For Ellipsoid Model, run the code in the second section of *Ellipsoid.m*.
