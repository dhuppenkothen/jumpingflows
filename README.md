# Reversible Jump MCMC with Normalizing Flows (maybe)

We have a problem where we have $K$ model components, and where the data is modelled as a linear sum of those $K$ components. We don't know what $K$ is and would like to infer it, and we also would like to infer parameters (and hyperparameters) at the same time. We can do this with Nested Sampling or MCMC, but can we do it with machine learning?

## Steps
* Is it possible to sample the joint distribution?
* IMplementing prior distributions and the likelihood in pytorch distributions
* Find a model where we can sample over $K$ that is
   * permutation invariant normalizing flow
   * but is conditional in hyperparameters $\alpha$
   * with a product over $K$


## Authors
* Ben Miller (GRAPPA/IvI)
* Daniela Huppenkothen (SRON)
