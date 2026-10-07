import numpy as np

def paired_bootstrap(values, n=10000, seed=0):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError("Invalid bootstrap observations")
    rng=np.random.default_rng(seed)
    means=np.concatenate([values[rng.integers(0,len(values),size=(min(500,n-i),len(values)))].mean(1) for i in range(0,n,500)])
    return {"estimate":float(values.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975)),"n_tasks":len(values)}

def paired_permutation(differences, n=10000, seed=0):
    differences=np.asarray(differences,float)
    observed=abs(float(differences.mean()))
    if not np.isfinite(differences).all(): raise ValueError("Invalid permutation observations")
    if len(differences)<=16:
        patterns=np.arange(2**len(differences),dtype=np.uint32)
        bits=((patterns[:,None]>>np.arange(len(differences)))&1)*2.-1.
        return float(np.mean(np.abs((bits*differences).mean(1))>=observed-1e-12))
    rng=np.random.default_rng(seed); exceed=0
    for i in range(0,n,500):
        signs=rng.choice([-1,1],size=(min(500,n-i),len(differences)))
        exceed+=int((np.abs((signs*differences).mean(1))>=observed-1e-12).sum())
    return (exceed+1)/(n+1)

def holm(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); result=np.zeros_like(p); running=0
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]); result[index]=min(running,1)
    return result.tolist()

def reliability(probabilities, labels, bins=10):
    p=np.asarray(probabilities,float); y=np.asarray(labels,float)
    if not len(p): return {"brier":None,"ece":None,"bins":[]}
    if np.any((p<0)|(p>1)): raise ValueError("Invalid probabilities")
    ids=np.minimum((p*bins).astype(int),bins-1); table=[]; ece=0
    for i in range(bins):
        selected=ids==i
        if not selected.any(): continue
        confidence=float(p[selected].mean()); observed=float(y[selected].mean()); n=int(selected.sum())
        table.append({"bin":i,"confidence":confidence,"observed":observed,"n":n})
        ece+=n/len(p)*abs(confidence-observed)
    return {"brier":float(np.mean((p-y)**2)),"ece":ece,"bins":table}

def pareto_front(points):
    points=sorted((float(c),float(q)) for c,q in points)
    frontier=[]; best=-np.inf
    for cost,quality in points:
        if frontier and cost==frontier[-1][0]:
            if quality>frontier[-1][1]: frontier[-1]=(cost,quality); best=quality
        elif quality>best: frontier.append((cost,quality)); best=quality
    return frontier

def pareto_auc(points, interval):
    lo,hi=interval
    if hi<=lo: return None
    front=pareto_front(points)
    if not front or lo<front[0][0] or hi>max(c for c,_ in points): return None
    x=np.array([c for c,q in front]); y=np.array([q for c,q in front])
    # Prespecified linear frontier interpolation, no extrapolation beyond observed raw range.
    interior=x[(x>lo)&(x<hi)]
    xs=np.r_[lo,interior,hi]; ys=np.interp(xs,x,y)
    return float(np.trapezoid(ys,xs)/(hi-lo))
