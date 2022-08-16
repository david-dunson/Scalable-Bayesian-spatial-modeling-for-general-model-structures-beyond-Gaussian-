## SiMPA

This package implements Simplified Manifold Preconditioner Adaptation in simple settings such as GLM models.

The algorithm is outlined in Section 3.2 of [*Spatial Meshing for General Bayesian Multivariate Models*](https://arxiv.org/abs/2201.10080).

R package [`meshed`](https://github.com/mkln/meshed) uses SiMPA with non-Gaussian outcomes. 

#### Sampling from custom densities
 - in `model_alts.h`, the class `MyDensity` can be used for custom densities. The two requirements are (1) a valid constructor, (2) a valid `compute_dens_grad_neghess` method which computes the density of the target, the gradient, and the negative hessian, at the input value `x`.
 - in `posterior.cpp`, make the appropriate changes for initializing `MyDensity`, then replace `GLMmodel` with `MyDensity`.
