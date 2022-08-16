#include <RcppArmadillo.h>
using namespace std;


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

// Gradient of the log posterior
inline arma::mat MyDensity::compute_dens_grad_neghess(
    double& loglike, arma::vec& grad_loglike, const arma::vec& x, 
    bool do_hess){
  
}

