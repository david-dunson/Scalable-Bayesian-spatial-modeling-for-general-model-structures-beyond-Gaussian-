library(simpa)
library(magrittr)
library(ggplot2)
library(dplyr)

#set.seed(1)
nr <- 50
p <- 6
beta <- rnorm(p)
X <- rnorm(p * nr) %>% matrix(ncol=p)
tau <- 1

#expmu <- 1/(1+exp( - X %*% beta ))
expmu <- exp( X %*% beta )
y <- #rbinom(nr, 1, expmu)
  rpois(nr, expmu)

# 0 gaussian
# 1 poisson
# 2 binomial

system.time({
  simpa_out <- posterior_sampling(y, X, tau, 1, 50000, 5000) 
})

beta_mcmc <- simpa_out$beta
eps_mcmc <- simpa_out$eps
M_mcmc <- simpa_out$M

M_mcmc[2,1,] %>% plot(type='l')
M_mcmc[2,1,] %>% tail(1000) %>% diff() %>% magrittr::equals(0) %>% mean()
eps_mcmc %>% plot(type='l')

beta_mcmc[2,] %>% #tail(5000) %>% 
  plot(type='l')
beta_mcmc %>% apply(1, \(x) coda::effectiveSize(tail(x, 2000)))
beta_mcmc %>% apply(1, mean)

glmobj <- glm(y ~ X-1, family="poisson")
stats::vcov(glmobj)


df_chain <- beta_mcmc %>% t() %>% as.data.frame()

df_chain %>% mutate(n=1:n()) %>% 
  head(50) %>% 
  ggplot(aes(V1, V2)) +
  geom_text(aes(label=n))
