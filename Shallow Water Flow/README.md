# Modeling of shallow-water flow, using shallow-water equations. #

The set of following equations represent the shallow water equations.

* δv<sub>x</sub> /δt = −v<sub>x</sub>(δv<sub>x</sub>/δx)−v<sub>y</sub> (δv<sub>x</sub>/δy) − g(δh/δx)

* δv<sub>y</sub>/δt = −v<sub>x</sub>(δv<sub>y</sub>/δx)−v<sub>y</sub>(δv<sub>y</sub>/δy)−g(δh/δy)
* δh/δt = −δ(v<sub>x</sub> \* h)/δx − δ(v<sub>y</sub> \* h)/δy

**Sixth-order accurate finite difference** method was used for spatial discretization of the equations. Then, **Fourth-order Runge-Kutta (RK4)** method was used to solve the equation in the temporal domain.

The resulting instruction set had code blocks which were embarrassingly parallel, for which Open-MP*(shared memory model)* code was developed. In addition Open-MPI(*distributed memory model*) code was developed for section of code which could be parallelized, but required message passing.