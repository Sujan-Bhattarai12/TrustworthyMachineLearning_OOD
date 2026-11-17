"""Build the final four figures and three tables from completed official evaluations."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAIN=Path('results/expanded_cifar10c_5000dev_2000test')
PANEL=Path('results/additional_corruption_smoke')
OUT=Path('results/reporting/final')
CLASSIFIERS=['resnet50','vit_b_16','swin_t']
CLASS_LABELS={'resnet50':'ResNet-50','vit_b_16':'ViT-B/16','swin_t':'Swin-T'}
METHODS=['IG dynamic','GRADCAM dynamic','MSP','ODIN','Entropy','Energy']
LABELS={'IG dynamic':'IG L/G/D + trends','GRADCAM dynamic':'Grad-CAM L/G/D + trends','MSP':'MSP','ODIN':'ODIN','Entropy':'Entropy','Energy':'Energy'}
COLORS=['#0072B2','#D55E00','#333333','#CC79A7','#009E73','#E69F00']
SOURCES=[]

def csv(path):
    SOURCES.append(Path(path));return pd.read_csv(path)

def js(path):
    SOURCES.append(Path(path));return json.loads(Path(path).read_text())

def table(frame,name):
    frame.to_csv(OUT/f'{name}.csv',index=False)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.3f}'
        return str(v)
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines+=['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    (OUT/f'{name}.md').write_text('\n'.join(lines)+'\n')
    frame.to_latex(OUT/f'{name}.tex',index=False,float_format='%.3f')

def export(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    all_rows=[];ablation_rows=[];cohorts=[];prevalence={}
    for c in CLASSIFIERS:
        primary=csv(MAIN/'evaluation'/c/'comparison.csv')
        supplementary=csv(MAIN/'supplementary_entropy_energy'/c/'comparison.csv')
        singles=csv(MAIN/'individual_lgd_ablations'/c/'comparison.csv')
        frozen=js(MAIN/'frozen'/c/'specifications.json');cohort=js(MAIN/'evaluation'/c/'cohort.json')
        trajectories=csv(MAIN/'evaluation'/c/'trajectories.csv')
        one=trajectories[trajectories.model=='IG dynamic']
        cohorts.append(dict(Classifier=CLASS_LABELS[c],Development_candidates=5000,Valid_development_images=len(frozen['fit_image_groups'])+len(frozen['calibration_image_groups']),
            Fitting_images=len(frozen['fit_image_groups']),Calibration_images=len(frozen['calibration_image_groups']),Test_candidates=2000,
            Valid_test_images=cohort['valid_trajectories'],Failing_trajectories=int(one.first_failure.notna().sum()),
            Nonfailing_trajectories=int(one.first_failure.isna().sum()),Failure_at_severity1=int((one.first_failure==1).sum())))
        combined=pd.concat([primary,supplementary,singles],ignore_index=True)
        combined['classifier']=c
        denominators=combined[['n_trajectories','n_failures','n_nonfailures']].drop_duplicates()
        if len(denominators)!=1:raise ValueError('Methods have different comparison cohorts.')
        all_rows.append(combined)
        prevalence[c]=float(primary.loc[primary.model=='MSP','prevalence'].iloc[0])
        for method in ['ig','gradcam']:
            prefix=method.upper()
            for label,model in [('L',f'{prefix} L alone'),('G',f'{prefix} G alone'),('D',f'{prefix} D alone'),
                                ('L/G/D',f'{prefix} static'),('L/G/D + trends',f'{prefix} dynamic'),('Interactions',f'{prefix} interactions')]:
                r=combined[combined.model==model].iloc[0]
                ablation_rows.append(dict(Classifier=CLASS_LABELS[c],Map='IG' if method=='ig' else 'Grad-CAM',Features=label,
                    Early_warning_pct=100*r.early_warning_rate,False_alarm_pct=100*r.false_alarm_rate,AUROC=r.auroc,Average_precision=r.aupr))
    data=pd.concat(all_rows,ignore_index=True);data.to_csv(OUT/'source_all_methods.csv',index=False)
    table(pd.DataFrame(cohorts),'table1_cohorts')
    main_rows=[]
    for c in CLASSIFIERS:
        for name in METHODS:
            r=data[(data.classifier==c)&(data.model==name)].iloc[0]
            main_rows.append(dict(Classifier=CLASS_LABELS[c],Method=LABELS[name],Early_warning_pct=100*r.early_warning_rate,
                Missed_warning_pct=100*r.missed_warning_rate,False_alarm_pct=100*r.false_alarm_rate,
                Mean_lead_steps=r.mean_lead_all_failures,AUROC=r.auroc,Average_precision=r.aupr))
    table(pd.DataFrame(main_rows),'table2_main_comparison')
    ablations=pd.DataFrame(ablation_rows);table(ablations,'table3_component_ablations')
    # Figure1: sensitivity and false alarms at independently development-calibrated thresholds.
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharex='row')
    for col,c in enumerate(CLASSIFIERS):
        ordered=data[data.classifier==c].set_index('model').loc[METHODS]
        for row,(metric,label) in enumerate([('early_warning_rate','Early warnings (%)'),('false_alarm_rate','False alarms (%)')]):
            ax=axes[row,col];values=100*ordered[metric].to_numpy()
            ax.barh(np.arange(6),values,color=COLORS);ax.set_yticks(np.arange(6),[LABELS[m] for m in METHODS] if col==0 else [])
            ax.invert_yaxis();ax.set_xlabel(label);ax.set_xlim(0,45);ax.grid(axis='x',alpha=.2)
            for y,v in enumerate(values):ax.text(v+.35,y,f'{v:.1f}',va='center',fontsize=9)
            if row==0:ax.set_title(CLASS_LABELS[c])
            else:ax.axvline(5,color='#777777',linestyle=':',linewidth=1)
    fig.suptitle('Frozen warning performance on official Gaussian blur',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));export(fig,'figure1_warning_and_false_alarms')
    # Figure2: next-step score ranking; not timing or probability-calibration measures.
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for col,c in enumerate(CLASSIFIERS):
        ordered=data[data.classifier==c].set_index('model').loc[METHODS]
        for row,(metric,label) in enumerate([('auroc','Next-step AUROC'),('aupr','Next-step average precision')]):
            ax=axes[row,col];values=ordered[metric].to_numpy()
            ax.barh(np.arange(6),values,color=COLORS);ax.set_yticks(np.arange(6),[LABELS[m] for m in METHODS] if col==0 else [])
            ax.invert_yaxis();ax.set_xlim(0,1.12);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
            ax.axvline(.5 if row==0 else prevalence[c],color='#777777',linestyle=':',linewidth=1)
            for y,v in enumerate(values):ax.text(v+.015,y,f'{v:.3f}',va='center',fontsize=8)
            if row==0:ax.set_title(CLASS_LABELS[c])
    fig.suptitle('Ranking next-step failures among eligible still-correct observations',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));export(fig,'figure2_next_step_discrimination')
    # Figure3: all singles plus the predeclared combined variants, no selecting a winner.
    columns=['L','G','D','L/G/D','L/G/D + trends','Interactions']
    labels=[f'{CLASS_LABELS[c]} / {m}' for c in CLASSIFIERS for m in ['IG','Grad-CAM']]
    fig,axes=plt.subplots(1,2,figsize=(16,5.6))
    for ax,metric,title in zip(axes,['Early_warning_pct','False_alarm_pct'],['Early warnings (%)','False alarms (%)']):
        matrix=np.array([[float(ablations[(ablations.Classifier==CLASS_LABELS[c])&(ablations.Map==m)&(ablations.Features==f)][metric].iloc[0])
            for f in columns] for c in CLASSIFIERS for m in ['IG','Grad-CAM']])
        image=ax.imshow(matrix,cmap='Blues' if metric=='Early_warning_pct' else 'Oranges',vmin=0,vmax=max(5,np.ceil(matrix.max()/5)*5),aspect='auto')
        ax.set_xticks(range(6),['L','G','D','L/G/D','+ trends','Interactions'],rotation=25,ha='right');ax.set_yticks(range(6),labels)
        ax.set_title(title)
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>.6*image.norm.vmax else 'black',fontsize=9)
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle('Individual attribution components do not consistently benefit from combination',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));export(fig,'figure3_attribution_ablations')
    panel=csv(PANEL/'comparison.csv');panel.to_csv(OUT/'supplement_corruption_panel.csv',index=False)
    policy='cross-corruption common images';corruptions=['shot_noise','motion_blur','fog','jpeg_compression'];markers=['o','s','^','D','P','X']
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    for ax,c in zip(axes.flat,corruptions):
        subset=panel[(panel.cohort_policy==policy)&(panel.corruption==c)].set_index('model')
        for method,color,marker in zip(METHODS,COLORS,markers):
            r=subset.loc[method];ax.scatter(100*r.false_alarm_rate,100*r.early_warning_rate,color=color,marker=marker,s=85,label=LABELS[method],edgecolor='white',linewidth=.5)
        r=subset.loc['MSP'];ax.set_title(f"{c.replace('_',' ').title()}\n{int(r.n_failures)} failing / {int(r.n_nonfailures)} nonfailing",fontsize=11)
        ax.set_xlim(-1,43);ax.set_ylim(0,33);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('False alarms (%) — lower is better')
    for ax in axes[:,0]:ax.set_ylabel('Early warnings (%) — higher is better')
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,-.01))
    fig.suptitle('Additional-corruption panel: frozen ResNet-50, same 187 images',fontsize=13)
    fig.tight_layout(rect=(0,.09,1,.94));export(fig,'figure4_corruption_generalization')
    SOURCES.extend([MAIN/'locked_protocol.json',PANEL/'protocol.json'])
    provenance=dict(data_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(SOURCES))},
        report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        main_scope='Expanded 5000/2000-candidate official corruption study, architecture-specific valid cohorts',
        supplements='Entropy/Energy and individual-component ablations added after main protocol; additional panel exploratory',
        aupr_definition='sklearn average_precision_score, not trapezoidal PR area',lead_definition='Severity steps; misses credited zero')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(f'Built four figures (PNG/PDF), three tables (CSV/Markdown/LaTeX) in {OUT}')

if __name__=='__main__':main()
