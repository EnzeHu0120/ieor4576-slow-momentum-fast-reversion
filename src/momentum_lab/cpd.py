"""Exact Matérn-3/2 GP changepoint detection from paper equations (4), (9), (10).

The state-space evaluation is algebraically equivalent to a dense GP likelihood,
but linear in window length. This CPU port avoids legacy TensorFlow/GPflow builds.
Every window is independently standardized and independently initialized at one.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import hashlib
import json
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from .paths import ROOT, configuration

@njit(cache=True, nogil=True)
def fast_likelihood(theta,y,change):
    """Scalar small-state Kalman algebra, equivalent to state_likelihood."""
    k=4 if change else 2
    A=np.zeros((k,2));S=np.zeros((k,k));Q=np.zeros((k,k))
    for block in range(k//2):
        i=2*block;var=np.exp(theta[0 if block==0 else 3]);q=np.sqrt(3.)/np.exp(theta[1 if block==0 else 4]);e=np.exp(-q)
        A[i,0]=e*(1+q);A[i,1]=e;A[i+1,0]=-e*q*q;A[i+1,1]=e*(1-q)
        S[i,i]=var;S[i+1,i+1]=var*q*q
        for ii in range(i,i+2):
            for jj in range(i,i+2):Q[ii,jj]=S[ii,jj]-A[ii,0]*S[i,i]*A[jj,0]-A[ii,1]*S[i+1,i+1]*A[jj,1]
    P=S.copy();mean=np.zeros(k);u=np.zeros(k);ph=np.zeros(k);PU=np.zeros((k,k))
    noise=np.exp(theta[2])+1e-8;value=0.
    for t in range(len(y)):
        g=1.;h=0.
        if change:
            z=max(-700.,min(700.,np.exp(theta[6])*(t-theta[5])));g=1/(1+np.exp(-z));h=1-g
        for i in range(k):ph[i]=P[i,0]*g+(P[i,2]*h if change else 0.)
        v=g*ph[0]+(h*ph[2] if change else 0.)+noise
        if v<=0 or not np.isfinite(v):return 1e30
        r=y[t]-g*mean[0]-(h*mean[2] if change else 0.)
        value+=.5*(np.log(2*np.pi*v)+r*r/v)
        for i in range(k):
            u[i]=mean[i]+ph[i]*r/v
            for j in range(k):PU[i,j]=P[i,j]-ph[i]*ph[j]/v
        for i in range(k):
            ib=(i//2)*2;mean[i]=A[i,0]*u[ib]+A[i,1]*u[ib+1]
            for j in range(k):
                jb=(j//2)*2
                P[i,j]=(A[i,0]*(A[j,0]*PU[ib,jb]+A[j,1]*PU[ib,jb+1])
                    +A[i,1]*(A[j,0]*PU[ib+1,jb]+A[j,1]*PU[ib+1,jb+1])+Q[i,j])
    return value

@njit(cache=True,nogil=True)
def fast_value_gradient(theta,y,change):
    value=fast_likelihood(theta,y,change);g=np.empty(len(theta))
    for j in range(len(theta)):
        up=theta.copy();down=theta.copy();up[j]+=1e-5;down[j]-=1e-5
        g[j]=(fast_likelihood(up,y,change)-fast_likelihood(down,y,change))/2e-5
    return value,g

@njit(cache=True)
def state_likelihood(theta, y, change):
    n=len(y); k=4 if change else 2
    a=np.exp(theta[0]); ell=np.exp(theta[1]); noise=np.exp(theta[2])
    a2=np.exp(theta[3]) if change else a
    ell2=np.exp(theta[4]) if change else ell
    loc=theta[5] if change else 0.
    steep=np.exp(theta[6]) if change else 1.
    transition=np.zeros((k,k)); stationary=np.zeros((k,k))
    for j in range(k//2):
        var=a if j==0 else a2; scale=ell if j==0 else ell2
        q=np.sqrt(3.)/scale; e=np.exp(-q); ix=2*j
        transition[ix,ix]=e*(1+q);transition[ix,ix+1]=e
        transition[ix+1,ix]=-e*q*q;transition[ix+1,ix+1]=e*(1-q)
        stationary[ix,ix]=var;stationary[ix+1,ix+1]=var*q*q
    process=stationary-transition@stationary@transition.T
    mean=np.zeros(k); covariance=stationary.copy(); loss=0.
    for t in range(n):
        h=np.zeros(k)
        if change:
            z=max(-700.,min(700.,steep*(t-loc)))
            gate=1./(1.+np.exp(-z));h[0]=gate;h[2]=1.-gate
        else:h[0]=1.
        ph=covariance@h; variance=h@ph+noise+1e-8
        if variance<=0 or not np.isfinite(variance):return 1e30
        residual=y[t]-h@mean
        loss+=.5*(np.log(2*np.pi*variance)+residual*residual/variance)
        mean+=ph*(residual/variance)
        covariance-=np.outer(ph,ph)/variance
        mean=transition@mean
        covariance=transition@covariance@transition.T+process
    return loss

@njit(cache=True)
def value_gradient(theta,y,change):
    # Forward sensitivities of the exact Kalman recursion. Differentiate the tiny
    # transition matrices once, rather than refitting a GP for each derivative.
    k=4 if change else 2; d=len(theta)
    A=np.zeros((k,k));S=np.zeros((k,k));dA=np.zeros((d,k,k));dS=np.zeros((d,k,k))
    for block in range(k//2):
        vi=0 if block==0 else 3;li=1 if block==0 else 4
        var=np.exp(theta[vi]);q=np.sqrt(3.)/np.exp(theta[li]);e=np.exp(-q);i=block*2
        A[i,i]=e*(1+q);A[i,i+1]=e;A[i+1,i]=-e*q*q;A[i+1,i+1]=e*(1-q)
        dA[li,i,i]=e*q*q;dA[li,i,i+1]=e*q
        dA[li,i+1,i]=e*(2*q*q-q*q*q);dA[li,i+1,i+1]=e*(2*q-q*q)
        S[i,i]=var;S[i+1,i+1]=var*q*q
        dS[vi,i,i]=var;dS[vi,i+1,i+1]=var*q*q;dS[li,i+1,i+1]=-2*var*q*q
    Q=S-A@S@A.T;dQ=np.empty_like(dS)
    for j in range(d):dQ[j]=dS[j]-dA[j]@S@A.T-A@dS[j]@A.T-A@S@dA[j].T
    noise=np.exp(theta[2]);mean=np.zeros(k);P=S.copy();dm=np.zeros((d,k));dP=dS.copy()
    value=0.;gradient=np.zeros(d)
    for t in range(len(y)):
        h=np.zeros(k);dh=np.zeros((d,k))
        if change:
            steep=np.exp(theta[6]);z=max(-700.,min(700.,steep*(t-theta[5])))
            gate=1/(1+np.exp(-z));h[0]=gate;h[2]=1-gate
            dh[5,0]=-steep*gate*(1-gate);dh[5,2]=-dh[5,0]
            dh[6,0]=steep*(t-theta[5])*gate*(1-gate);dh[6,2]=-dh[6,0]
        else:h[0]=1.
        ph=P@h;v=h@ph+noise+1e-8;r=y[t]-h@mean
        if v<=0 or not np.isfinite(v):return 1e30,np.zeros(d)
        value+=.5*(np.log(2*np.pi*v)+r*r/v)
        newdm=np.empty_like(dm);newdp=np.empty_like(dP)
        updated=mean+ph*(r/v);updatedP=P-np.outer(ph,ph)/v
        for j in range(d):
            dp=dP[j]@h+P@dh[j]
            dv=dh[j]@ph+h@dp+(noise if j==2 else 0.)
            dr=-dh[j]@mean-h@dm[j]
            gradient[j]+=.5*((1/v-r*r/(v*v))*dv+2*r/v*dr)
            um=dm[j]+dp*(r/v)+ph*(dr/v-r*dv/(v*v))
            up=dP[j]-(np.outer(dp,ph)+np.outer(ph,dp))/v+np.outer(ph,ph)*dv/(v*v)
            newdm[j]=dA[j]@updated+A@um
            newdp[j]=dA[j]@updatedP@A.T+A@up@A.T+A@updatedP@dA[j].T+dQ[j]
        mean=A@updated;P=A@updatedP@A.T+Q;dm=newdm;dP=newdp
    return value,gradient

def fit_window(y,maxiter=200):
    y=np.asarray(y,dtype=np.float64); sd=y.std()
    if sd<1e-12: return (0.5,0.5,0.,0.,0,True)
    y=(y-y.mean())/sd;l=len(y)-1
    bounds=[(-12.,12.)]*3
    objective=fast_value_gradient
    m=minimize(objective,np.zeros(3),args=(y,False),jac=True,
        method="L-BFGS-B",bounds=bounds, options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5})
    initial=np.array([m.x[0],m.x[1],m.x[2],m.x[0],m.x[1],l/2,0.])
    b=bounds+[(-12.,12.),(-12.,12.),(1e-5,l-1e-5),(-9.,9.)]
    cp=minimize(objective,initial,args=(y,True),jac=True,method="L-BFGS-B",
        bounds=b,options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5})
    return (float(expit(m.fun-cp.fun)),float((l-cp.x[5])/l),float(m.fun),
        float(cp.fun),int(m.nit+cp.nit),bool(m.success and cp.success))

def dense_value_gradient(theta,y,change):
    n=len(y);x=np.arange(n,dtype=float);distance=np.abs(x[:,None]-x[None,:])
    def matern(v,l):
        z=np.sqrt(3)*distance/np.exp(l); ez=np.exp(-z); a=np.exp(v)
        return a*(1+z)*ez,a*z*z*ez
    k1,dl1=matern(theta[0],theta[1]);noise=np.exp(theta[2])
    if change:
        k2,dl2=matern(theta[3],theta[4]);s=np.exp(theta[6]);g=expit(s*(x-theta[5]));h=1-g
        gg=np.outer(g,g);hh=np.outer(h,h);cov=k1*gg+k2*hh
        dc=-s*g*h;ds=s*(x-theta[5])*g*h
        def dg(z):return k1*(np.outer(z,g)+np.outer(g,z))-k2*(np.outer(z,h)+np.outer(h,z))
        derivatives=[k1*gg,dl1*gg,noise*np.eye(n),k2*hh,dl2*hh,dg(dc),dg(ds)]
    else:
        cov=k1.copy();derivatives=[k1,dl1,noise*np.eye(n)]
    cov.flat[::n+1]+=noise+1e-8
    factor=cho_factor(cov,lower=True,check_finite=False);alpha=cho_solve(factor,y,check_finite=False)
    loss=.5*y@alpha+np.log(np.diag(factor[0])).sum()+.5*n*np.log(2*np.pi)
    b=cho_solve(factor,np.eye(n),check_finite=False)-np.outer(alpha,alpha)
    gradient=.5*np.einsum('ij,kij->k',b,np.asarray(derivatives))
    return loss,gradient

def compute_asset(job):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    ticker,lbw,maxiter=job
    path=ROOT/f"data/processed/cpd/{lbw}/{ticker}.parquet";path.parent.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(ROOT/"data/processed/features.parquet",filters=[("ticker","=",ticker)])
    f=f.sort_values("date")
    # Start CPD using all returns before feature burnin as in the author module.
    rolled=pd.read_parquet(ROOT/f"data/interim/{ticker}_rolls.parquet")
    # Author CPD uses returns of the winsorized series. Recreate the pre-feature history.
    r=rolled.raw_return.dropna();close=100*(1+r).cumprod()
    ewm=close.ewm(halflife=252);srs=close.clip(ewm.mean()-5*ewm.std(),ewm.mean()+5*ewm.std())
    returns=(srs/srs.shift(1)-1).dropna()
    dates=set(f.date)
    old=pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done=set(old.date) if len(old) else set()
    rows=[];start=time.monotonic()
    y=returns.to_numpy(dtype=float)
    for i in range(lbw,len(returns)):
        d=returns.index[i]
        if d not in dates or d in done:continue
        score,loc,m,cp,nit,success=fit_window(y[i-lbw:i+1],maxiter)
        rows.append({"date":d,"ticker":ticker,f"cp_score_{lbw}":score,
            f"cp_rl_{lbw}":loc,"matern_nlml":m,"changepoint_nlml":cp,
            "iterations":nit,"converged":success})
        if len(rows)%1000==0:
            pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_parquet(path,index=False)
    if rows:pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_parquet(path,index=False)
    return ticker,lbw,len(rows),round(time.monotonic()-start,2)

def run(config_path="configs/config.yaml",windows=None):
    cfg=configuration(config_path)
    root=ROOT/"data/processed/cpd";root.mkdir(parents=True,exist_ok=True)
    signature={"features_sha256":hashlib.sha256((ROOT/"data/processed/features.parquet").read_bytes()).hexdigest(),
        "maxiter":cfg["cpd_maxiter"],"source_sha256":hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    version=root/"version.json"
    if version.exists() and json.loads(version.read_text())!=signature:
        raise RuntimeError("CPD cache belongs to different data/settings/source; preserve it elsewhere before rebuilding")
    version.write_text(json.dumps(signature,indent=2))
    f=pd.read_parquet(ROOT/"data/processed/features.parquet",columns=["ticker"])
    tickers=sorted(f.ticker.unique());windows=windows or cfg["cpd_windows"]
    jobs=[(t,l,cfg["cpd_maxiter"]) for l in windows for t in tickers]
    # Compile before spawning; each worker then loads its own persistent numba cache.
    state_likelihood(np.zeros(3),np.ones(12),False)
    fast_value_gradient(np.array([0.,0.,0.,0.,0.,5.,0.]),np.ones(12),True)
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures=[pool.submit(compute_asset,j) for j in jobs]
        for future in as_completed(futures):print("GP-CPD",*future.result(),flush=True)
