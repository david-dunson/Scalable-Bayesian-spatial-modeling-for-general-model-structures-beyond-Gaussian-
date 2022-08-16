#define ARMA_DONT_PRINT_ERRORS


#ifndef SIMPA_STEP
#define SIMPA_STEP

#include "simpa_adapt.h"

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