"""Publication figures from retained raw outcomes and fresh audit artifacts."""
import json,pathlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/mive_revision/figures';OUT.mkdir(parents=True,exist_ok=True)
ART=ROOT/'experiments/mive_revision/artifacts'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})

def save(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+suffix),dpi=400)
    plt.close(fig)

def workflow():
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(7,3.1));ax.set(xlim=(0,10),ylim=(0,4.6));ax.axis('off')
    def box(x,y,w,h,text,color='#edf3f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.08',fc=color,ec='#34536b',lw=.9))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#34536b',lw=.9))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=7)
    ax.text(.1,4.3,'A   Independent continuation measurement',weight='bold')
    box(.1,2.5,1.6,1.1,'Restore full state\nVerify SHA256\nFix feasible actions')
    box(2.25,2.5,1.55,1.1,'Force one action\nTHINK / RETRIEVE\nVERIFY / STOP')
    box(4.4,3.15,2.0,.75,'Labeling μ\nIndependent A/B sets')
    box(4.4,1.95,2.0,.75,'Deployment π\nIndependent A/B sets')
    box(7.05,2.4,2.8,1.35,'Score terminal answers\nCharge all future resources\nCompare values and ranks\nBootstrap by question')
    arrow((1.7,3.05),(2.25,3.05));arrow((3.8,3.05),(4.4,3.5));arrow((3.8,3.05),(4.4,2.35))
    arrow((6.4,3.5),(7.05,3.1));arrow((6.4,2.35),(7.05,2.8))
    ax.text(.1,1.4,'B   Separate coupling audit',weight='bold')
    box(.1,.25,4.2,.85,'Shared sampler seeds across action branches\nSame checkpoints and K per arm', '#f1f6ed')
    box(5.15,.25,4.7,.85,'Compare THINK action-contrast variance\nwith independent seeds at equal branch counts', '#f1f6ed')
    arrow((4.3,.68),(5.15,.68))
    fig.tight_layout();save(fig,'workflow')

def archival():
    rows=json.loads((ART/'archival_reanalysis.json').read_text())
    colors={'CRAVE':'#d55e00','confidence':'#0072b2','fixed':'#009e73','llm_router':'#7a5195'}
    fig,axes=plt.subplots(1,2,figsize=(7,3.1),sharey=True)
    for ax,dataset in zip(axes,['HotpotQA','2WikiMultiHopQA']):
        for row in rows:
            if row['dataset']!=dataset:continue
            x=row['metrics']['generated_tokens'];y=row['metrics']['exact_match']
            ax.errorbar(x['mean'],y['mean'],xerr=np.array([[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]]),
                        yerr=np.array([[y['mean']-y['ci95'][0]],[y['ci95'][1]-y['mean']]]),fmt='o',ms=5,
                        color=colors[row['method']],capsize=2,label=row['method'],alpha=.85)
        ax.set(title=dataset,xlabel='Mean generated tokens',ylim=(0,1));ax.grid(alpha=.15)
    axes[0].set_ylabel('Normalized answer EM');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'archival_quality_cost')

def audit():
    data=json.loads((ART/'dev4b/checkpoint_metrics.json').read_text())
    utility=json.loads((ART/'dev4b/utility_checkpoint_metrics.json').read_text())
    actions=['THINK','RETRIEVE','VERIFY','STOP']
    fig,axes=plt.subplots(1,2,figsize=(7,max(3.5,.23*len(data)+1)),sharey=True)
    labels=[r['dataset'].replace('MultiHopQA','')+' '+str(i+1) for i,r in enumerate(data)]
    for ax,field,title in zip(axes,['action_values_mu','action_values_pi'],['Labeling continuation μ','Legacy deployment π']):
        values=np.array([[r[field].get(a,np.nan) for a in actions] for r in data])
        im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap='cividis',aspect='auto')
        ax.set(xticks=range(4),xticklabels=actions,yticks=range(len(data)),yticklabels=labels,title=title)
        for (i,j),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=7,color='white' if v<.45 else 'black')
    fig.subplots_adjust(left=.18,right=.88,wspace=.1,bottom=.14,top=.88)
    color_axis=fig.add_axes([.91,.14,.018,.74])
    fig.colorbar(im,cax=color_axis,label='Mean terminal EM');save(fig,'continuation_values')
    primary=json.loads((ART/'dev4b/summary.json').read_text());secondary=json.loads((ART/'dev4b/utility_summary.json').read_text())
    metrics=[('signed_THINK_bias','Signed THINK bias'),('absolute_discrepancy','Absolute discrepancy'),('argmax_change','Argmax change'),('THINK_STOP_rank_reversal','THINK STOP reversal')]
    fig,axes=plt.subplots(1,2,figsize=(7,3.2),sharey=True)
    for ax,summary,title in zip(axes,[primary,secondary],['Primary terminal EM','Secondary priced value']):
        for i,(metric,label) in enumerate(metrics):
            row=summary[metric]
            if row['estimate'] is None:continue
            y=len(metrics)-1-i;m=row['estimate'];lo,hi=row['ci95']
            ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#0072b2',capsize=3)
        ax.axvline(0,color='gray',lw=.8);ax.set(title=title,yticks=range(4),yticklabels=[x[1] for x in metrics[::-1]],xlabel='Estimate (95% question CI)');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'mismatch_intervals')
    fig,axes=plt.subplots(1,2,figsize=(7,3))
    for i,(field,label) in enumerate([('observed_squared_discrepancy','Observed squared gap'),('estimated_noise','Estimated sampling noise'),('estimated_squared_shift','Unbiased squared shift')]):
        x=primary[field];m=x['estimate'];lo,hi=x['ci95'];axes[0].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color='#0072b2')
    axes[0].set(xticks=range(3),xticklabels=['Observed','Noise','Shift'],ylabel='Mean squared discrepancy');axes[0].axhline(0,color='gray',lw=.8)
    x=np.array([r['independent_gap_variance'] for r in utility if r['independent_gap_variance'] is not None]);y=np.array([r['matched_gap_variance'] for r in utility if r['matched_gap_variance'] is not None])
    colors={'HotpotQA':'#0072b2','2WikiMultiHopQA':'#d55e00','MuSiQue':'#009e73'}
    for dataset,color in colors.items():
        points=[r for r in utility if r['dataset']==dataset and r['independent_gap_variance'] is not None]
        axes[1].scatter([r['independent_gap_variance'] for r in points],[r['matched_gap_variance'] for r in points],c=color,s=25,label=dataset,alpha=.8)
    axes[1].legend(frameon=False,fontsize=6.5)
    limit=max(x.max(),y.max(),.001)*1.1;axes[1].plot([0,limit],[0,limit],color='gray',ls='--',lw=.8)
    axes[1].set(xlim=(-limit*.03,limit),ylim=(-limit*.03,limit),xlabel='Independent seed contrast variance',ylabel='Matched seed contrast variance')
    fig.tight_layout();save(fig,'noise_and_coupling')

if __name__=='__main__':workflow();archival();audit()
