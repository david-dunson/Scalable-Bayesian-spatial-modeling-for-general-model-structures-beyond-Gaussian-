library(simpa)
library(magrittr)
library(ggplot2)
library(dplyr)

set.seed(1)
nr <- 50
p <- 6
beta <- rnorm(p)

# make correlated covariates
xgrid <- runif(p*2) %>% matrix(ncol=2)
X <- mvtnorm::rmvnorm(nr, rep(0, p), exp(-as.matrix(dist(xgrid))))
# make uncorrelated covariates
# X <- rnorm(p * nr) %>% matrix(ncol=p)

#expmu <- 1/(1+exp( - X %*% beta ))
expmu <- exp( X %*% beta )
y <- #rbinom(nr, 1, expmu)
  rpois(nr, expmu)

# 0 gaussian
# 1 poisson
# 2 binomial

system.time({
  simpa_out <- posterior_sampling(y, X, 1, 1, 5000, 500) 
})

beta_mcmc <- simpa_out$beta
eps_mcmc <- simpa_out$eps
M_mcmc <- simpa_out$M

# show how preconditioner is being adapted
M_mcmc[2,1,] %>% plot(type='l')
# step size adaptation via dual averaging
eps_mcmc %>% plot(type='l')
# Markov chains for regression coefficients
df <- beta_mcmc %>% t() %>% as.data.frame() %>%
  mutate(m = 1:n()) %>%
  tidyr::gather(variable, chain, -m)

ggplot(df, aes(m, chain)) +
  geom_line() + 
  facet_wrap(~ variable, scales="free") +
  theme_minimal()
