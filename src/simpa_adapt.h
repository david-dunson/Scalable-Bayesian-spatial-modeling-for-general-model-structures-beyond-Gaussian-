#define ARMA_DONT_PRINT_ERRORS

#ifndef SIMPA_ADAPT 
#define SIMPA_ADAPT


#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

class AdaptSimpa {
public:
  int i;
  
  int n;
  double mu;
  double eps;
  double eps_bar;
  double H_bar;
  double gamma;
  double t0;
  double kappa;
  int M_adapt;
  double delta;
  
  double alpha;
  double n_alpha;
  
  bool adapt_C;
 
  arma::mat C_const;
  arma::mat Ccholinv_const;
  
  AdaptSimpa();
  AdaptSimpa(double, int, bool, int);
  
  // count iterations
  void step();
  
  // determine if adapting
  bool preconditioner_adapting(double);
  
  // adapt eps
  void eps_adapt_step();
  
  
  void preconditioner_propose_adapt(arma::mat&);
  void preconditioner_update(const arma::mat&, 
                      const arma::mat&);
  
};


inline AdaptSimpa::AdaptSimpa(){
  
}

inline AdaptSimpa::AdaptSimpa(double eps0, int size, bool adapt_preconditioner=true, int M_adapt_in=0){
  i = 0;
  mu = log(10 * eps0);
  eps = eps0;
  eps_bar = eps0; //M_adapt_in == 0? eps0 : 1;
  H_bar = 0;
  gamma = .05;//.05;
  t0 = 10;
  kappa = 0.75;
  delta = 0.575; // target accept
  M_adapt = M_adapt_in; // default is no adaptation of eps
  
  alpha = 0;
  n_alpha = 0;
  
  adapt_C = adapt_preconditioner;
  n = size;

  if(adapt_C){
    C_const = arma::eye(n, n);
    //Cinv_const = C_const;
    Ccholinv_const = arma::eye(n, n);
  }
}

inline bool AdaptSimpa::preconditioner_adapting(double ru){
  //return adapt_C & (i>i_C_adapt);
  // outside of initial burn period AND not randomly adapting
  if(adapt_C){
    double T = 500.0;
    double kappa = 0.7;// 0.33;
    double irate = std::min(1.0, (i+.0)/10000) * kappa;
    return (i <= T) | (ru < pow(i-T, -irate));  
  } else {
    return false;
  }
  
}

inline void AdaptSimpa::preconditioner_update(const arma::mat& MM, 
                                      const arma::mat& Mcholinv){
  C_const = MM;
  Ccholinv_const = Mcholinv;
}

inline void AdaptSimpa::preconditioner_propose_adapt(arma::mat& MM){
  if(adapt_C){
    double gamma = 1.0/10.0;
    MM = C_const + gamma * (MM - C_const);  
  }
}

inline void AdaptSimpa::step(){
  i++;
}

//inline bool AdaptSimpa::eps_adapting(){
//  return (i < M_adapt);
//}

inline void AdaptSimpa::eps_adapt_step(){
  int m = i+1; //i<i_C_adapt? 0 : i+1-i_C_adapt;
  if(m < M_adapt){ // ***
    
    H_bar = (1.0 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
    eps = exp(mu - sqrt(m)/gamma * H_bar);
    //Rcpp::Rcout << m+t0 << " " << delta << "  " << alpha/n_alpha << " " << eps << endl;
    eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
    //Rcpp::Rcout << "eps: " << eps << ", eps_bar: " << eps_bar << " | alpha: " << alpha << ", n_alpha: " << n_alpha << "\n";
  } else {
    eps = eps_bar;
  }
}


#endif
