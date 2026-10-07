from collections import defaultdict
import numpy as np
from crave.agent.state import State, ACTIONS
from crave.counterfactual.branch import interaction_targets
from crave.evaluation.task_metrics import utility

def arrays(records, features, lambdas, references, ablation=None):
    grouped=defaultdict(list)
    for r in records:
        if r["split"] != "train": raise ValueError("Training rejects non-train branch labels")
        grouped[r["state_hash"]].append(r)
    xs,ys,ms,its,ims=[],[],[],[],[]
    for state_hash,rows in sorted(grouped.items()):
        state=State.from_dict(rows[0]["state_before"])
        interaction=interaction_targets(rows,lambdas,references)
        for weight in lambdas:
            target=np.zeros((4,6),dtype=np.float32); mask=np.zeros(4,dtype=np.float32)
            for ai,action in enumerate(ACTIONS):
                selected=[r for r in rows if r["forced_sequence"]==[action]]
                if not selected: continue
                mask[ai]=1
                rewards=[r["immediate"]["reward"] if ablation=="-RTG" else r["reward"] for r in selected]
                costs=[r["immediate"]["cost"] if ablation=="-RTG" else r["future_cost"] for r in selected]
                target[ai,0]=np.mean([utility(r,c,weight,references) for r,c in zip(rewards,costs)])
                target[ai,1]=np.mean(rewards)
                target[ai,2:]=np.mean(costs,axis=0)/np.array(references)
            synergy=np.zeros(16,dtype=np.float32); imask=np.zeros(16,dtype=np.float32)
            if ablation not in ["-Interaction","-RTG","-CF"]:
                for (pair,w),value in interaction.items():
                    if w==weight:
                        idx=ACTIONS.index(pair[0])*4+ACTIONS.index(pair[1]); synergy[idx]=value; imask[idx]=1
            xs.append(features(state,weight)); ys.append(target); ms.append(mask); its.append(synergy); ims.append(imask)
    if not xs: raise ValueError("No valid training states")
    return tuple(np.stack(items) for items in [xs,ys,ms,its,ims])
