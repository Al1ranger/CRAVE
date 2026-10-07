"""Additional manuscript figures reconstructed from frozen observations."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'crave/runs/expanded_8484164aef776a12'
OUT=ROOT/'tmp/full_manuscript_assets';OUT.mkdir(parents=True,exist_ok=True)
episodes=json.loads((RUN/'evaluation.json').read_text())['data']
with open(RUN/'results/statistics.csv',encoding='utf-8') as f:stats=list(csv.DictReader(f))
methods=['CRAVE','confidence','fixed','llm_router'];names=['CRAVE','Confidence','Fixed','LLM router']
datasets=['HotpotQA','2WikiMultiHopQA'];short=['HotpotQA','2Wiki']
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)

# Schematic is an algorithm diagram, with no simulated empirical data.
fig,ax=plt.subplots(figsize=(6.8,4.0));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,label,color='#EAF2F8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color,edgecolor='#34495E',linewidth=.9))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.0)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.0,'color':'#34495E'})
ax.text(.1,5.65,'a  Offline matched interventions',fontweight='bold',fontsize=10)
box(.15,3.95,2.0,1.05,'Saved public state\nSame hash\nSame budget')
for i,a in enumerate(['THINK','RETRIEVE','VERIFY','STOP']):
    y=5.0-i*.62;box(2.75,y-.35,1.25,.47,a)
    arrow((2.15,4.48),(2.75,y-.11));arrow((4.06,y-.11),(4.75,4.48))
box(4.8,3.95,2.05,1.05,'Shared continuation\nμ₀ with K = 3\nMatched seeds')
box(7.5,3.95,2.25,1.05,'Offline outcome\nand future cost\nQ and pair contrasts',color='#E8F6EF')
arrow((6.9,4.48),(7.44,4.48))
ax.text(.1,2.4,'b  Online controller',fontweight='bold',fontsize=9)
box(.15,1.0,2.15,1.05,'Query and summaries\nEvidence and answer\nRemaining budget')
box(2.95,1.0,2.0,1.05,'Frozen MiniLM\n1549 state features')
box(5.6,1.0,1.8,1.05,'512 → 256 MLP\nQ, success, cost\nInteraction heads',color='#E8F6EF')
box(8.0,1.0,1.8,1.05,'Feasibility mask\nEligible Q argmax\nExecute action')
for a,b in [((2.36,1.52),(2.9,1.52)),((5.0,1.52),(5.55,1.52)),((7.45,1.52),(7.95,1.52))]:arrow(a,b)
arrow((8.65,3.9),(6.5,2.1))
ax.text(8.95,3.0,'Supervised learning',fontsize=8,ha='center')
ax.plot([8.9,8.9,1.22,1.22],[.95,.5,.5,.95],color='#34495E',lw=.9)
arrow((1.22,.5),(1.22,.95))
ax.text(5.0,.08,'Update public state and remaining resources until STOP',ha='center',fontsize=8.5)
save(fig,'method_workflow')

# Mean action counts; bootstrap questions after averaging the three seeds.
action_rows=[];rng=np.random.default_rng(20261007)
fig,axes=plt.subplots(1,2,figsize=(6.8,3.0),sharey=True)
action_names=['THINK','RETRIEVE','VERIFY','STOP'];palette=['#0072B2','#009E73','#D55E00','#6C757D']
for di,dataset in enumerate(datasets):
    for mi,method in enumerate(methods):
        selected=[e for e in episodes if (e['dataset'],e['method'])==(dataset,method)]
        grouped=defaultdict(list)
        for e in selected:grouped[e['task_id']].append(Counter(a['action'] for a in e['actions']))
        vals=np.array([[np.mean([c[a] for c in seeds]) for a in action_names] for seeds in grouped.values()])
        assert vals.shape==(20,4)
        boot=vals[rng.integers(0,20,(10000,20))].mean(axis=1)
        for ai,a in enumerate(action_names):
            mean=vals[:,ai].mean();lo,hi=np.quantile(boot[:,ai],[.025,.975]);y=3-mi+(ai-1.5)*.17
            axes[di].errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=3,color=palette[ai],capsize=2,label=a if mi==0 else None)
            action_rows.append({'dataset':dataset,'method':method,'action':a,'mean_count':mean,'ci_low':lo,'ci_high':hi})
    axes[di].set(title=short[di],xlabel='Actions per episode',xlim=(-.15,7.8),yticks=range(4),yticklabels=names[::-1],ylim=(-.6,3.6));axes[di].grid(axis='x',alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,.1,1,1));save(fig,'action_usage')

# Reconcile the four implemented utility-cost terms exactly for every episode.
components=[('Output / 4096','generated_tokens',4096),('Retrieval / 4','retrieval_calls',4),('Verification / 2','verification_calls',2),('Logical seconds / 60','latency',60)]
cost_rows=[];fig,axes=plt.subplots(1,2,figsize=(6.8,3.0),sharex=True)
for e in episodes:assert abs(sum(e[k]/scale for _,k,scale in components)-e['normalized_cost'])<1e-10
for di,dataset in enumerate(datasets):
    for mi,method in enumerate(methods):
        es=[e for e in episodes if (e['dataset'],e['method'])==(dataset,method)];left=0
        for ci,(label,key,scale) in enumerate(components):
            value=np.mean([e[key]/scale for e in es]);axes[di].barh(3-mi,value,left=left,color=palette[ci],label=label if mi==0 else None)
            cost_rows.append({'dataset':dataset,'method':method,'component':key,'reference_scale':scale,'mean_normalized_component':value});left+=value
        axes[di].text(left+.01,3-mi,f'{left:.3f}',va='center',fontsize=8)
    axes[di].set(title=short[di],xlabel='Normalized composite cost',yticks=range(4),yticklabels=names[::-1],xlim=(0,1.09));axes[di].grid(axis='x',alpha=.12)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,.16,1,1));save(fig,'resource_decomposition')

# Unstandardized paired effects preserve each metric's units.
fig,axes=plt.subplots(2,3,figsize=(6.8,4.0))
comparators=['confidence','fixed','llm_router'];metric_names=['exact_match','generated_tokens','normalized_cost']
for di,dataset in enumerate(datasets):
    for ci,metric in enumerate(metric_names):
        ax=axes[di,ci];scale=100 if metric=='exact_match' else 1
        for mi,other in enumerate(comparators):
            r=next(r for r in stats if r['dataset']==dataset and r['comparison']=='CRAVE minus '+other and r['metric']==metric)
            value,lo,hi=[float(r[k])*scale for k in ['estimate','ci_low','ci_high']]
            ax.errorbar(value,2-mi,xerr=[[value-lo],[hi-value]],fmt='o',color=colors[mi+1],capsize=3,ms=4)
        ax.axvline(0,color='#333333',linestyle='--',lw=.8);ax.set_yticks(range(3),['Router','Fixed','Confidence'])
        ax.set_ylim(-.5,2.5);ax.grid(axis='x',alpha=.12)
        ax.set(title=short[di],xlabel=['EM percentage points','Output tokens','Normalized cost'][ci])
fig.tight_layout(w_pad=1.2,h_pad=1.4);save(fig,'paired_effect_intervals')
for name,rows in [('action_usage.csv',action_rows),('resource_decomposition.csv',cost_rows)]:
    with open(OUT/name,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'episodes':len(episodes),'question_clusters':40,'added_figures':4,'cost_components_reconcile_all_episodes':True,'source_config_hash':json.loads((RUN/'FROZEN_EXPANDED_CONFIG.json').read_text())['sha256']}
(OUT/'scientific_figures_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
