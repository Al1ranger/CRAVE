"""Small two-hidden-layer MLP with explicit analytic gradients and Adam.

NumPy keeps controller training independent of the GPU inference runtime.
Heads: 4*(Q, success logit, four future-cost logits), plus 16 ordered synergies.
"""
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def softplus(x): return np.logaddexp(0, x)
def huber(x): return np.where(np.abs(x) <= 1, 0.5*x*x, np.abs(x)-0.5)
def huber_grad(x): return np.clip(x, -1, 1)

class ValueModel:
    def __init__(self, inputs, hidden=(512,256), seed=0):
        rng = np.random.default_rng(seed)
        dims = [inputs, *hidden, 40]
        self.params = {}
        for i, (a,b) in enumerate(zip(dims, dims[1:])):
            self.params[f"W{i}"] = (rng.standard_normal((a,b))*np.sqrt(2/a)).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(b, dtype=np.float32)
        self.hidden = hidden

    def forward(self, x):
        z1 = x @ self.params["W0"] + self.params["b0"]
        h1 = np.maximum(z1,0)
        z2 = h1 @ self.params["W1"] + self.params["b1"]
        h2 = np.maximum(z2,0)
        output = h2 @ self.params["W2"] + self.params["b2"]
        return output, (x,z1,h1,z2,h2)

    def predict(self, x): return self.forward(x)[0]

    @staticmethod
    def loss(output, target, mask, interactions, interaction_mask, weights, tau=0.1):
        n = max(float(mask.sum()),1)
        heads = output[:,:24].reshape(-1,4,6)
        gradient = np.zeros_like(output)
        g = gradient[:,:24].reshape(-1,4,6)
        q, p, c = heads[:,:,0], sigmoid(heads[:,:,1]), softplus(heads[:,:,2:])
        dq, dp, dc = q-target[:,:,0], p-target[:,:,1], c-target[:,:,2:]
        loss = float((huber(dq)*mask).sum()/n)
        g[:,:,0] = huber_grad(dq)*mask/n
        loss += weights["cal"]*float((dp*dp*mask).sum()/n)
        g[:,:,1] = weights["cal"]*2*dp*p*(1-p)*mask/n
        loss += weights["cost"]*float((huber(dc)*mask[:,:,None]).sum()/(4*n))
        g[:,:,2:] = weights["cost"]*huber_grad(dc)*sigmoid(heads[:,:,2:])*mask[:,:,None]/(4*n)
        pair_count = 0
        pairs = []
        for a in range(4):
            for b in range(a+1,4):
                valid = mask[:,a]*mask[:,b]*(np.abs(target[:,a,0]-target[:,b,0])>1e-6)
                pair_count += valid.sum()
                pairs.append((a,b,valid))
        pair_count = max(float(pair_count),1)
        for a,b,valid in pairs:
            difference = (q[:,a]-q[:,b])/tau
            y = (target[:,a,0] > target[:,b,0]).astype(float)
            loss += weights["rank"]*float(((np.logaddexp(0,difference)-y*difference)*valid).sum()/pair_count)
            pair_gradient = weights["rank"]*(sigmoid(difference)-y)*valid/(tau*pair_count)
            g[:,a,0] += pair_gradient
            g[:,b,0] -= pair_gradient
        ni = max(float(interaction_mask.sum()),1)
        difference = output[:,24:]-interactions
        loss += weights["interaction"]*float((huber(difference)*interaction_mask).sum()/ni)
        gradient[:,24:] = weights["interaction"]*huber_grad(difference)*interaction_mask/ni
        return loss, gradient

    def gradients(self, cache, output_gradient):
        x,z1,h1,z2,h2 = cache
        g = {}
        g["W2"], g["b2"] = h2.T @ output_gradient, output_gradient.sum(0)
        dz2 = (output_gradient @ self.params["W2"].T)*(z2>0)
        g["W1"], g["b1"] = h1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ self.params["W1"].T)*(z1>0)
        g["W0"], g["b0"] = x.T @ dz1, dz1.sum(0)
        return g

    def fit(self, x, target, mask, interactions, interaction_mask, weights=None, epochs=100, lr=1e-3, seed=0, batch_size=64):
        weights = weights or {"cost":0.25,"cal":0.25,"rank":0.25,"interaction":0.25}
        rng = np.random.default_rng(seed)
        first = {k:np.zeros_like(v) for k,v in self.params.items()}
        second = {k:np.zeros_like(v) for k,v in self.params.items()}
        losses, step = [], 0
        for epoch in range(epochs):
            order = rng.permutation(len(x)); epoch_losses=[]
            for offset in range(0,len(x),batch_size):
                idx = order[offset:offset+batch_size]
                output,cache = self.forward(x[idx])
                loss,gradient = self.loss(output,target[idx],mask[idx],interactions[idx],interaction_mask[idx],weights)
                if not np.isfinite(loss): raise ValueError("Non-finite training loss")
                gradients = self.gradients(cache,gradient)
                norm = np.sqrt(sum(float((g*g).sum()) for g in gradients.values()))
                scale = min(1, 5/max(norm,1e-12)); step += 1
                for k,g in gradients.items():
                    g = g*scale
                    first[k] = 0.9*first[k]+0.1*g
                    second[k] = 0.999*second[k]+0.001*g*g
                    self.params[k] -= lr*(first[k]/(1-0.9**step))/(np.sqrt(second[k]/(1-0.999**step))+1e-8)
                epoch_losses.append(loss)
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def save(self,path): np.savez_compressed(path,**self.params)

    @classmethod
    def load(cls,path):
        weights=np.load(path)
        model=cls(weights["W0"].shape[0],(weights["W0"].shape[1],weights["W1"].shape[1]))
        model.params={k:weights[k] for k in weights.files}
        return model
