"""Apply frozen warning scores to reserved blur trajectories. No fitting/calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_early_warnings import prepare, trajectory_outcomes, TRAJECTORY
from scripts.freeze_warning_methods import predict


def evaluate(artifact_path, input_path, output_dir):
    artifact=json.loads(Path(artifact_path).read_text())
    df=pd.read_csv(input_path)
    if set(df.source)!={'test'} or set(df.corruption)!={'blur'} or set(df.classifier)!={artifact['classifier']}:
        raise ValueError('Final input must be reserved test-source blur for the frozen classifier.')
    for key,value in artifact['settings'].items():
        if not np.allclose(df[key],value):
            raise ValueError(f'Frozen collection setting mismatch: {key}')
    manifest=pd.read_csv('splits/split_manifest_v2.csv')
    allowed=set(manifest.loc[(manifest.split=='test')&(manifest.source=='test'),'image_id'])
    if not set(df.image_id)<=allowed:
        raise ValueError('Images are not in the reserved final-test manifest.')
    data,excluded=prepare(df)
    outcomes=[]
    for spec in artifact['methods']:
        result=trajectory_outcomes(data,predict(spec,data),spec['threshold'])
        result['model']=spec['name']
        result['threshold']=spec['threshold']
        outcomes.append(result)
    trajectories=pd.concat(outcomes,ignore_index=True)
    summary=[]
    for model,group in trajectories.groupby('model'):
        failed=group[group.first_failure.notna()]
        negatives=group[group.first_failure.isna()]
        summary.append(dict(classifier=artifact['classifier'],model=model,n_trajectories=len(group),
                            n_failures=len(failed),n_nonfailures=len(negatives),
                            n_early_warnings=int(failed.early_warning.sum()),
                            n_missed_warnings=int(failed.missed_warning.sum()),
                            early_warning_rate=float(failed.early_warning.mean()) if len(failed) else np.nan,
                            mean_lead_all_failures=float(failed.credited_lead_steps.mean()) if len(failed) else np.nan,
                            mean_lead_detected_only=float(failed.detected_lead_steps.mean()),
                            false_alarm_rate=float(negatives.false_alarm.mean()) if len(negatives) else np.nan))
    pairs=[]
    proposed=trajectories[~trajectories.model.isin(['MSP','ODIN'])]
    for baseline in ['MSP','ODIN']:
        both=proposed.merge(trajectories[trajectories.model==baseline],on=TRAJECTORY,suffixes=('','_baseline'),validate='one_to_one')
        both=both[both.first_failure.notna()]
        delta=both.credited_lead_steps-both.credited_lead_steps_baseline
        pairs.append(dict(baseline=baseline,n_shared_failures=len(both),n_earlier=int((delta>0).sum()),
                          n_tied=int((delta==0).sum()),n_later=int((delta<0).sum()),
                          mean_credited_lead_difference=float(delta.mean())))
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in [('summary',pd.DataFrame(summary)),('trajectories',trajectories),
                       ('paired_comparisons',pd.DataFrame(pairs)),('excluded_trajectories',excluded)]:
        frame.to_csv(output_dir/f'{name}.csv',index=False)
    (output_dir/'frozen_artifact_used.json').write_text(json.dumps(dict(
        path=str(artifact_path),sha256=hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        fitted_on_test=False,thresholds_retuned=False,artifact=artifact),indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.artifact,args.input,args.output_dir)


if __name__=='__main__':
    main()
