"""Convex regularized logistic ERM, group-balanced ERM, and group DRO.

Group DRO minimizes max_g mean logistic loss_g + l2/2 ||w||² via an epigraph
constraint, not a heuristic sample-weighting approximation. Bias is unpenalized.
Feature scaling fits fitting data only. Group labels never enter predictors.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from src.warning_features import validate_attribution_columns


def risk_and_gradient(theta,x,y):
    logits=x@theta[:-1]+theta[-1]
    residual=expit(logits)-y
    return float(np.mean(np.logaddexp(0,logits)-y*logits)),np.r_[x.T@residual/len(y),residual.mean()]


def fit_logistic_objective(x,y,groups,objective='erm',l2=.01,beta=1.):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);groups=np.asarray(groups)
    if objective not in ['erm','balanced_erm','group_dro','vrex'] or l2<=0 or beta<0:
        raise ValueError('Unknown objective or nonpositive regularization.')
    if x.ndim!=2 or len(x)!=len(y) or len(y)!=len(groups) or not len(y):
        raise ValueError('Invalid fitting arrays.')
    if not np.isfinite(x).all() or set(np.unique(y))!={0.,1.}:
        raise ValueError('Fitting requires finite features and both binary labels.')
    scaler=StandardScaler().fit(x)
    scaled=scaler.transform(x)
    masks=[groups==group for group in np.unique(groups)]
    initial=np.zeros(scaled.shape[1]+1)
    def penalty(theta):return .5*l2*np.dot(theta[:-1],theta[:-1])
    def penalty_grad(theta):return np.r_[l2*theta[:-1],0.]
    def empirical(theta):
        if objective in ['balanced_erm','vrex']:
            results=[risk_and_gradient(theta,scaled[mask],y[mask]) for mask in masks]
            risks=np.array([r[0] for r in results]);gradients=np.array([r[1] for r in results])
            risk=risks.mean();gradient=gradients.mean(axis=0)
            if objective=='vrex':
                risk+=beta*np.var(risks)
                gradient+=2*beta*np.mean((risks-risks.mean())[:,None]*gradients,axis=0)
        else:risk,gradient=risk_and_gradient(theta,scaled,y)
        return risk+penalty(theta),gradient+penalty_grad(theta)
    if objective=='group_dro':
        def value(z):return z[-1]+penalty(z[:-1])
        def gradient(z):return np.r_[penalty_grad(z[:-1]),1.]
        def constraints(z):
            return np.asarray([z[-1]-risk_and_gradient(z[:-1],scaled[m],y[m])[0] for m in masks])
        def constraint_gradient(z):
            return np.asarray([np.r_[-risk_and_gradient(z[:-1],scaled[m],y[m])[1],1.] for m in masks])
        start=np.r_[initial,np.log(2.)]
        result=minimize(value,start,jac=gradient,method='SLSQP',
            constraints=[dict(type='ineq',fun=constraints,jac=constraint_gradient)],options=dict(maxiter=1000,ftol=1e-10))
        theta=result.x[:-1]
        if not result.success or np.min(constraints(result.x)) < -1e-6:
            raise ValueError(f'Group DRO solver failed: {result.message}')
    else:
        result=minimize(empirical,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7))
        theta=result.x
        if not result.success:raise ValueError(f'ERM solver failed: {result.message}')
    risks={str(group):risk_and_gradient(theta,scaled[groups==group],y[groups==group])[0] for group in np.unique(groups)}
    return dict(objective=objective,l2=l2,beta=beta,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        weights=theta[:-1].tolist(),intercept=float(theta[-1]),training_group_risks=risks,
        worst_training_group_risk=max(risks.values()),penalty=float(penalty(theta)),
        n_fit=len(y),solver_success=bool(result.success),solver_iterations=int(result.nit))


def fit_attribution_warning(frame,columns,objective='erm',l2=.01,beta=1.):
    validate_attribution_columns(columns)
    model=fit_logistic_objective(frame[columns],frame.fails_next_step,frame.corruption,objective,l2,beta)
    return dict(model,columns=list(columns),role='attribution')


def probability(model,frame):
    if model.get('role')=='attribution':validate_attribution_columns(model['columns'])
    x=(frame[model['columns']].to_numpy()-np.asarray(model['mean']))/np.asarray(model['scale'])
    return expit(x@np.asarray(model['weights'])+model['intercept'])
