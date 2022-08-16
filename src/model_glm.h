#ifndef UTILS_DENS_GRAD 
#define UTILS_DENS_GRAD

#include <RcppArmadillo.h>

using namespace std;

const double TOL_LOG_LOW=exp(-15);
const double TOL_HIGH=exp(25);
const double TOL_LOG_HIGH=25;

inline double gaussian_logdensity(const double& x, const double& sigsq){
  return -0.5*log(2.0 * M_PI * sigsq) -0.5/sigsq * x*x;
}

inline double gaussian_loggradient(const double& x, const double& sigsq){
  // derivative wrt mean parameter
  return x/sigsq;
}

inline double poisson_logpmf(const double& x, double lambda){
  if(lambda < TOL_LOG_LOW){
    lambda = TOL_LOG_LOW;
  } else {
    if(lambda > TOL_HIGH){
      lambda = TOL_HIGH;
    }
  }
  return x * log(lambda) - lambda - lgamma(x+1);
}

inline double poisson_loggradient(const double& y, const double& offset, const double& w){
  // llik: y * log(lambda) - lambda - lgamma(y+1);
  // lambda = exp(o + w);
  // llik: y * (o + w) - exp(o+w);
  // grad: y - exp(o+w)
  if(offset + w > TOL_LOG_HIGH){
    return y - TOL_HIGH;
  }
  return y - exp(offset + w);
}

inline double poisson_neghess_mult_sqrt(const double& mu){
  return pow(mu, 0.5);
}

inline double bernoulli_logpmf(const double& x, double p){
  if(p > 1-TOL_LOG_LOW){
    p = 1-TOL_LOG_LOW;
  } else {
    if(p < TOL_LOG_LOW){
      p = TOL_LOG_LOW;
    }
  }
  return x * log(p) + (1-x) * log(1-p);
}

inline double bernoulli_loggradient(const double& y, const double& offset, const double& w){
  // llik: (y-1) * (o+w) - log{1+exp(-o-w)}
  // grad: y-1 + exp(-o-w)/(1+exp(-o-w))
  return y-1 + 1.0/(1.0+exp(offset+w));
}

inline double bernoulli_neghess_mult_sqrt(const double& exij){
  double opexij = (1.0 + exij);
  return pow(exij / (opexij*opexij), 0.5);
}

inline double betareg_logdens(const double& y, const double& mu, double phi){
  // ferrari & cribari-neto A3
  // using logistic link
  double muphi = mu*phi;
  return R::lgammafn(phi) - R::lgammafn(muphi) - R::lgammafn(phi - muphi) +
    (muphi - 1.0) * log(y) + 
    (phi - muphi - 1.0) * log(1.0-y);
  
}

inline double betareg_loggradient(const double& ystar, const double& mu, const double& phi){
  // ferrari & cribari-neto A3
  // using logistic link
  double muphi = mu*phi;
  double oneminusmu = 1.0-mu;
  //double ystar = log(y/(1.0-y));
  double mustar = R::digamma(muphi) - R::digamma(phi - muphi);
  
  return phi * (ystar - mustar) * mu * oneminusmu;
}

inline double betareg_neghess_mult_sqrt(const double& sigmoid, const double& tausq){
  double tausq2 = tausq * tausq;
  return pow(- 1.0/tausq2 * (R::trigamma( sigmoid / tausq ) + 
    R::trigamma( (1.0-sigmoid) / tausq ) ) * pow(sigmoid * (1.0 - sigmoid), 2.0), .5);  // notation of 
}

inline double negbin_logdens(const double& y, double mu, double logmu, double alpha){
  // Cameron & Trivedi 2013 p. 81
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
    logmu = TOL_LOG_HIGH;
  }
  if(alpha < TOL_LOG_LOW){
    // reverts to poisson
    return y * logmu - mu - lgamma(y+1);
  } 
  double sumj = 0;
  for(int j=0; j<y; j++){
    sumj += log(j + 1.0/alpha);
  }
  double p = 1.0 + alpha * mu;
  return sumj - lgamma(y+1) - (y+1.0/alpha) * log(p) + y * (log(alpha) + logmu);
}

inline double negbin_loggradient(const double& y, double mu, const double& alpha){
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
  }
  return ((y-mu) / (1.0 + alpha * mu));
}

inline double negbin_neghess_mult_sqrt(const double& y, double logmu, const double& alpha){
  double mu = exp(logmu);
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
    logmu = TOL_LOG_HIGH;
  }
  double onealphamu = (1.0 + alpha*mu);
  double result = pow(mu/onealphamu, .5);
  return result;
}

inline double get_mult(const double& y, const double& tausq, const double& offset, 
                       const double& xij, const int& family){
  // if m is the output from this function, then
  // m^2 X'X is the negative hessian of a glm model in which X*x is the linear term
  double mult=1;
  if(family == 0){  // family=="gaussian"
    mult = pow(tausq, -0.5);
  } else if (family == 1){
    double mu = exp(offset + xij);
    mult = poisson_neghess_mult_sqrt(mu);
  } else if (family == 2){
    double exij = exp(- offset - xij);
    mult = bernoulli_neghess_mult_sqrt(exij);
  } else if (family == 3){
    double sigmoid = 1.0/(1.0 + exp(-offset - xij));
    mult = betareg_neghess_mult_sqrt(sigmoid, tausq);
  } else if(family == 4){
    double logmu = offset + xij;
    double alpha = tausq;
    mult = negbin_neghess_mult_sqrt(y, logmu, alpha);
  }
  return mult;
}

class GLMmodel {
public:
  arma::vec y;
  arma::mat X;
  double tau;
  int family;
  
  arma::vec ones;

  int n;
  int p;
  
  arma::mat compute_dens_grad_neghess(double& loglike, arma::vec& grad_loglike, 
                                      const arma::vec& x, 
                                      bool do_hess=true);
  
  GLMmodel(const arma::vec& yin,
           const arma::mat& Xin,
           double tauin,
           int familyin);
  
};

GLMmodel::GLMmodel(const arma::vec& yin,
         const arma::mat& Xin,
         double tauin,
         int familyin){
  
  n = yin.n_elem;
  p = Xin.n_cols;
  
  y = yin;
  X = Xin;
  tau = tauin;
  family = familyin;
  
  ones = arma::ones(n);
}

// Gradient of the log posterior
inline arma::mat GLMmodel::compute_dens_grad_neghess(
    double& loglike, arma::vec& grad_loglike, const arma::vec& x, 
    bool do_hess){
  
  arma::vec Xx = X*x;
  arma::vec muvec = arma::zeros(p);
  arma::vec offset = arma::zeros(n);
  
  arma::mat XtX = arma::zeros(1,1);
  if(do_hess || (family==0)){
    arma::mat Xresult = X;
    arma::vec mult = arma::zeros(n);
    for(unsigned int i=0; i<X.n_rows; i++){
      mult(i) = get_mult(y(i), tau, offset(i), Xx(i), family);
      Xresult.row(i) = X.row(i) * mult(i);
    }
    XtX = Xresult.t() * Xresult + arma::eye(p, p);
  }
  
  if(family==0){ // gaussian
    arma::mat Xres = X.t() * y;
    loglike = 1.0/tau * arma::conv_to<double>::from(
      Xres.t() * x - .5 * x.t() * XtX * x);
    grad_loglike = 1.0/tau * (Xres - XtX * x);
    
  } else if(family == 1){ // poisson
    arma::vec muvec = arma::zeros(n);
    for(int i=0; i<n; i++){
      double logmu = Xx(i);
      if(logmu > TOL_LOG_HIGH){
        logmu = TOL_LOG_HIGH;
      }
      muvec(i) = exp(logmu);
    }
    loglike = arma::conv_to<double>::from( y.t() * Xx - ones.t() * muvec );
    //Rcpp::Rcout << "yXx:  " << y.t() * Xx << " " <<  ones.t() * muvec << endl;
    grad_loglike = X.t() * (y - muvec);
    
  } else if(family == 2){ // binomial
    // x=beta
    arma::vec muvec = 1.0/(1.0 + exp(-Xx));
    // y and y1 are both zero when missing data
    loglike = arma::conv_to<double>::from( 
      y.t() * log(muvec + 1e-6) + (1-y).t() * log(1-muvec + 1e-6) );
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
    }
    grad_loglike = X.t() * (y - muvec);
    
  } else if(family == 4){ // negative binomial
    arma::vec logcomps = arma::zeros(n);
    for(int i=0; i<n; i++){
      double logmu = Xx(i);
      if(logmu > TOL_LOG_HIGH){
        logmu = TOL_LOG_HIGH;
      }
      double mu = exp(logmu);
      logcomps(i) = negbin_logdens(y(i), mu, logmu, tau);
      muvec(i) = exp(logmu);
    }
    loglike += arma::accu(logcomps);
  }
  return XtX;
}

#endif
