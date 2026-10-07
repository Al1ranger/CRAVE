"""Task-level views of the frozen expanded test; no parameter tuning."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
RUN=Path(json.loads((BASE/'crave/artifacts/expanded_test/LATEST_RUN.json').read_text())['run_directory'])
OUT=RUN/'results/figures';OUT.mkdir(exist_ok=True)
def read(name):
    with open(RUN/'results'/name,encoding='utf-8') as f:return list(csv.DictReader(f))
summary=read('summary.csv');tasks=read('task_means.csv')
lookup={(r['dataset'],r['method']):r for r in summary}
methods=['CRAVE','confidence','fixed','llm_router'];names=['CRAVE','Confidence','Fixed','LLM router']
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
datasets=['HotpotQA','2WikiMultiHopQA'];short=['HotpotQA','2Wiki']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig,axes=plt.subplots(2,2,figsize=(6.8,5.1))
rng=np.random.default_rng(20261005)
for di,dataset in enumerate(datasets):
    for mi,method in enumerate(methods):
        selected=[r for r in tasks if (r['dataset'],r['method'])==(dataset,method)]
        y=3-mi
        jitter=rng.uniform(-.12,.12,len(selected))
        for ci,metric in enumerate(['exact_match','generated_tokens']):
            ax=axes[di,ci];r=lookup[(dataset,method)]
            vals=np.array([float(t[metric]) for t in selected]);mean=float(r[metric]);lo=float(r[metric+'_ci_low']);hi=float(r[metric+'_ci_high'])
            scale=100 if metric=='exact_match' else 1
            ax.scatter(vals*scale,y+jitter,s=15,alpha=.34,color=colors[mi],edgecolors='none',zorder=2)
            ax.errorbar(mean*scale,y,xerr=[[(mean-lo)*scale],[(hi-mean)*scale]],fmt='D',color=colors[mi],capsize=3,ms=5,zorder=3)
            ax.set_yticks(range(4),names[::-1]);ax.set_ylim(-.45,3.45)
            ax.grid(axis='x',alpha=.15,zorder=0)
    axes[di,0].set(xlabel='Exact match percent',xlim=(-4,104),title=f'{short[di]}  20 questions')
    axes[di,1].set(xlabel='Generated output tokens',xlim=(-12,524),title=f'{short[di]}  Per-question costs')
fig.tight_layout(h_pad=1.7,w_pad=1.2)
for ext in ['png','pdf']:fig.savefig(OUT/('expanded_quality_cost.'+ext),bbox_inches='tight')
plt.close(fig)

means={(r['dataset'],r['task_id'],r['method']):r for r in tasks}
mapping=[]
fig,axes=plt.subplots(1,2,figsize=(6.8,5.2),sharex=True)
palette={'CRAVE higher EM':'#009E73','Equal EM':'#6C757D','CRAVE lower EM':'#D55E00'}
for di,dataset in enumerate(datasets):
    ids=sorted(set(r['task_id'] for r in tasks if r['dataset']==dataset))
    pairs=[]
    for task_id in ids:
        a=means[(dataset,task_id,'CRAVE')];b=means[(dataset,task_id,'confidence')]
        delta=float(a['generated_tokens'])-float(b['generated_tokens']);em=float(a['exact_match'])-float(b['exact_match'])
        group='CRAVE higher EM' if em>1e-12 else ('CRAVE lower EM' if em<-1e-12 else 'Equal EM')
        pairs.append((delta,task_id,em,group))
    pairs.sort()
    labels=[]
    for i,(delta,task_id,em,group) in enumerate(pairs):
        label=('H' if di==0 else 'W')+f'{i+1:02}'
        labels.append(label)
        axes[di].barh(i,delta,color=palette[group],height=.7)
        mapping.append({'dataset':dataset,'chart_label':label,'task_id':task_id,'output_token_difference_crave_minus_confidence':delta,'exact_match_difference':em,'outcome':group})
    axes[di].axvline(0,color='black',lw=.8);axes[di].set_yticks(range(20),labels,fontsize=7.5)
    axes[di].set(title=f'{short[di]}  20 paired questions',xlabel='CRAVE minus confidence output tokens',ylabel='Question ordered by token difference')
    axes[di].grid(axis='x',alpha=.15)
handles=[plt.Rectangle((0,0),1,1,color=color) for color in palette.values()]
fig.legend(handles,list(palette),loc='lower center',ncol=3,fontsize=8,frameon=False)
fig.tight_layout(rect=(0,.06,1,1),w_pad=1.4)
for ext in ['png','pdf']:fig.savefig(OUT/('paired_question_costs.'+ext),bbox_inches='tight')
plt.close(fig)
with open(OUT/'paired_question_key.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(mapping[0]));writer.writeheader();writer.writerows(mapping)
print(json.dumps({'figures':2,'task_clusters':40,'run':str(RUN)}))
