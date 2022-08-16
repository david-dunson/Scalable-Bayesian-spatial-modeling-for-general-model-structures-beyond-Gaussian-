#include <RcppArmadillo.h>
using namespace std;

/*
 * Use this file to sample from custom densities
 */

class MyDensity {
public:
  int d;
  
  arma::mat compute_dens_grad_neghess(double& loglike, arma::vec& grad_loglike, 
                                      const arma::vec& x, 
                                      bool do_hess=true);
  
  MyDensity();
  
};

MyDensity::MyDensity(){
  
  
}

inline arma::mat MyDensity::compute_dens_grad_neghess(
    double& loglike, arma::vec& grad_loglike, const arma::vec& x, 
    bool do_hess){
  // loglike = ...
  // grad_loglike = ...
  // if(do_hess){
  //  return neg_hessian 
  // } else { 
  //  return arma::zeros(1,1);
  // }
  
}

