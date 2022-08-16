#define ARMA_DONT_PRINT_ERRORS

#include "model_glm.h"
#include "simpa.h"

//[[Rcpp::export]]
Rcpp::List posterior_sampling(const arma::vec& y, 
                              const arma::mat& X,
                              double tau=1, int family=0,
                              int mcmc = 100, int print_every=100){

  int p = X.n_cols;
  arma::mat beta_mcmc = arma::zeros(p, mcmc);
  
  arma::vec beta = arma::zeros(p);
  
  arma::cube M_mcmc = arma::zeros(p,p,mcmc);
  arma::vec eps_mcmc = arma::zeros(mcmc);
  
  double eps = 1;
  
  AdaptSimpa beta_adapt(eps, p, true, 10000);
  
  GLMmodel model(y, X, tau, family);
  
  for(int m = 0; m<mcmc; m++){
    //Rcpp::Rcout << beta_adapt.eps << endl;
    beta = simpa_step(beta, model, beta_adapt, false);
    
    beta_mcmc.col(m) = beta;
    eps_mcmc(m) = beta_adapt.eps;
    M_mcmc.slice(m) = beta_adapt.C_const;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "m: " <<  m << " " << beta_adapt.eps << endl;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_mcmc,
    Rcpp::Named("eps") = eps_mcmc,
    Rcpp::Named("M") = M_mcmc
  );
  
}
