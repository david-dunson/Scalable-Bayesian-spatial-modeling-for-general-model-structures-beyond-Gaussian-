#define ARMA_DONT_PRINT_ERRORS

#ifndef SIMPA 
#define SIMPA


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
  eps_bar = eps0;
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
    Ccholinv_const = arma::eye(n, n);
  }
}

inline bool AdaptSimpa::preconditioner_adapting(double ru){
  if(adapt_C){
    double T = 500.0;
    double kappa = 0.7;
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

inline void AdaptSimpa::eps_adapt_step(){
  int m = i+1; 
  if(m < M_adapt){ 
    H_bar = (1.0 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
    eps = exp(mu - sqrt(m)/gamma * H_bar);
    eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
  } else {
    eps = eps_bar;
  }
}

template <class T>
inline arma::vec simpa_step(arma::vec current_x, 
                            T& model,
                            AdaptSimpa& adaptparams, 
                            bool debug=false){
  // with infinite adaptation
  int n = current_x.n_elem;
  // currents
  arma::vec xgrad;
  double joint0, eps1, eps2;
  
  arma::mat MM, Minvchol;
  
  adaptparams.n_alpha = 1.0;
  bool fwd_chol_error = false;
  bool rev_chol_error = false;
  
  adaptparams.step();
  
  // diminishing adaptation: probability of updating eps & preconditioner 
  double runifadapt = arma::conv_to<double>::from(arma::randu(1));
  bool adapt_now = adaptparams.preconditioner_adapting(runifadapt);
  
  if(adapt_now) {
    // adapting at this time; 
    
    //joint0 = compute_dens(current_x);
    //xgrad = compute_grad(current_x);
    //arma::mat MM = compute_neghess(current_x);
    MM = model.compute_dens_grad_neghess(joint0, xgrad, current_x, true);
    if(MM.has_inf()){
      fwd_chol_error = true;
    } else {
      try {
        MM = MM/MM(0,0);
        adaptparams.preconditioner_propose_adapt(MM);
        Minvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(MM), "lower")));
      } catch (...) {
        fwd_chol_error = true;
      }
    }
  } else {
    // not adapting at this time
    MM = model.compute_dens_grad_neghess(joint0, xgrad, current_x, false);
    MM = adaptparams.C_const;
    Minvchol = adaptparams.Ccholinv_const;
  }
  
  eps1 = adaptparams.eps;
  eps2 = eps1 * eps1;
  
  if(fwd_chol_error || xgrad.has_nan() || xgrad.has_inf() || std::isnan(joint0) || std::isinf(joint0)){
    adaptparams.alpha = 0.0;
    adaptparams.eps_adapt_step();
    return current_x;
  }
  
  arma::vec proposal_mean = current_x + eps2 * 0.5 * Minvchol.t() * Minvchol * xgrad;
  
  // proposal value
  arma::vec p = arma::randn(n);
  arma::vec q = proposal_mean + eps1 * Minvchol.t() * p;
  
  // proposal
  double joint1;
  arma::vec revgrad;
  
  arma::mat RR, Rinvchol;
  
  if(adapt_now) {
    // initial burn period use full riemann manifold
    //joint1 = compute_dens(q);
    //revgrad = compute_grad(q);
    //arma::mat RR = compute_neghess(q);
    RR = model.compute_dens_grad_neghess(joint1, revgrad, q, true);
    
    if(RR.has_inf()){
      rev_chol_error = true;
    } else {
      try {
        RR = RR/RR(0,0);
        adaptparams.preconditioner_propose_adapt(RR);
        Rinvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(RR), "lower")));
      } catch (...) {
        rev_chol_error = true;
      }
    }
  } else {
    // after burn period keep constant variance
    //joint1 = compute_dens(q);
    //revgrad = compute_grad(q);
    RR = model.compute_dens_grad_neghess(joint1, revgrad, q, false);
    RR = adaptparams.C_const;
    Rinvchol = adaptparams.Ccholinv_const;
  }
  
  if(rev_chol_error || revgrad.has_inf() || std::isnan(joint1) || std::isinf(joint1)){
    adaptparams.alpha = 0.0;
    adaptparams.eps_adapt_step();
    return current_x;
  }
  
  double Richoldet = arma::accu(log(Rinvchol.diag()));
  double Micholdet = arma::accu(log(Minvchol.diag()));
  
  arma::vec reverse_mean = q + eps2 * 0.5 * Rinvchol.t() * Rinvchol * revgrad; 
  
  double prop0to1 = Micholdet -.5/eps2 * arma::conv_to<double>::from(
    (q - proposal_mean).t() * MM * (q - proposal_mean) );
  double prop1to0 = Richoldet -.5/eps2 * arma::conv_to<double>::from(
    (current_x - reverse_mean).t() * RR * (current_x - reverse_mean) );
  
  adaptparams.alpha = std::min(1.0, exp(joint1 + prop1to0 - joint0 - prop0to1));
  
  double accepting_u = arma::randu();
  if(accepting_u < adaptparams.alpha){ 
    current_x = q;
    if(adapt_now){
      // accepted: send accepted covariance to adaptation
      adaptparams.preconditioner_update(RR, Rinvchol); 
    }
  } else {
    if(adapt_now){
      // rejected: send original covariance to adaptation
      adaptparams.preconditioner_update(MM, Minvchol);
    }
  }
  adaptparams.eps_adapt_step();
  return current_x;
}



#endif
