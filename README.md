# NUMERICAL METHODS : Solving Ordinary and Partial Differential Equations #

This repository contains solution to partial differential equations, corresponding to the following two problems in the domain of mechanics.

# 1.  Modeling of shallow-water flow, using shallow-water equations. #
**REPORT :** [link](https://github.com/anubhav-cs/Numerical-Methods/blob/master/Shallow%20Water%20Flow/Modelling%20the%20Shallow%20Water%20Flow.pdf)

The set of following equations represent the shallow water equations.

* δv<sub>x</sub> /δt = −v<sub>x</sub>(δv<sub>x</sub>/δx)−v<sub>y</sub> (δv<sub>x</sub>/δy) − g(δh/δx)

* δv<sub>y</sub>/δt = −v<sub>x</sub>(δv<sub>y</sub>/δx)−v<sub>y</sub>(δv<sub>y</sub>/δy)−g(δh/δy)
* δh/δt = −δ(v<sub>x</sub> \* h)/δx − δ(v<sub>y</sub> \* h)/δy

**Sixth-order accurate finite difference** method was used for spatial discretization of the equations. Then, **Fourth-order Runge-Kutta (RK4)** method was used to solve the equation in the temporal domain.

The resulting instruction set had code blocks which were embarrassingly parallel, for which Open-MP*(shared memory model)* code was developed. In addition Open-MPI(*distributed memory model*) code was developed for section of code which could be parallelized, but required message passing.

# 2. Measuring Temperature variation as a function of time in a 3D heat sink. #
**REPORT**: [link](https://github.com/anubhav-cs/Numerical-Methods/blob/master/Temperature%20variation%20in%20a%20Heat%20Sink/TheHeatEquation_HeatSink.pdf)

* The Heat Equation:

  The heat equation is of the form of a PDE, which derives from the principle of conservation of energy.

  ρ C (∂T/∂t) = k ∇<sup>2</sup>T


* Problem structure and solution using Numerical Methods:

  An unstructured grid made up of 3D tetrahedron elements was used as the model for heat sink. The heat equation is a type of partial differential equation(PDE) which can solved using numerical methods. To solve the equation, the **Galerkin method of weighted residuals** was applied on the finite elements in the spatial domain and the **Implicit Euler method** was used to perform the time marching. The objective was to design a system to study temperature variation(over-time and spatially) in the 3D-models, given particular environmental settings (boundary conditions and heat sink material etc...). In this particular case, a constant heat flux(from CPU or other heat source) from the bottom and ambient air surrounding the heat sink on all other sides was assumed. The heat sink was assumed to be of copper which provided values for thermal conductivity, mass density and specific heat capacity.


* The *quest* for **parallelization**:

  OpenCL parallization
  The OpenCL library was used to develop parallel code for two problems, ‘daxpy’ (**Y =**a**X+ Y**) and ‘inner product’ (**Y = X<sup>T</sup> X**) . This could be used in the conjugate gradient method where there are few calculations which are performed at every time-step to solve the system of PDE, **A φ<sup>l+1</sup> = b** , which can be parallelized with above code.
  The following equation represent the values which are calculated at each time-step in conjugate gradient method.

  1. φ<sup>k+1</sup> = φ<sup>k</sup> + α d<sup>k</sup>
  2. r<sup>k+1</sup> = r<sup>k</sup>  − α d<sup>k</sup>  ,
  3. β = ((r<sup>k+1</sup>)<sup>T</sup>r<sup>k+1</sup>)/((r<sup>k</sup>)<sup>T</sup>r<sup>k</sup>)
  4. d<sup>k+1</sup> = r<sup>k+1</sup> + β d<sup>k</sup>

  **The 1<sup>st</sup>, 2<sup>nd</sup> and 4<sup>th</sup> equation can be parallelized using the ‘daxpy’ method, where as both the numerator and denominator of 3<sup>rd</sup> equation can be calculated used the inner-product method.** Each of the parameters here are 1D-vector stored in an array of size <number of points>. The parallelization is beneficial here since these can be very large vectors with hundred thousands of points. This is specially true for ‘daxpy’ method since the entire calculation can be parallelized in one step. However, for the inner product the product of individual items has to be summed up as well, which leads to some amount of serialization in execution of the OpenCL Code.
