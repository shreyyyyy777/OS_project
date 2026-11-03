from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dashboard'))
from utils import load_experiment_artifacts,live_snapshot,mode_metrics
st.set_page_config(page_title='BGOpt Research Dashboard',page_icon='B',layout='wide')
st.title('BGOpt research dashboard')
st.caption('Experiment 001 results and a strictly read-only telemetry monitor')
page=st.sidebar.radio('Navigate',['Overview','Live Monitor','Experiment 001','ML Analysis','Run Project'])
@st.cache_data
def artifacts(): return load_experiment_artifacts(ROOT)
rows,summary=artifacts()
if page=='Overview':
 st.header('Safe background-process optimisation, evaluated transparently')
 st.write('BGOpt characterises workload behaviour and combines it with host pressure and protection rules to propose conservative priority changes. This dashboard never executes a proposal.')
 st.markdown('**Telemetry -> workload characterisation -> pressure + protection -> policy -> suggested action**')
 c=st.columns(4); c[0].metric('Experiment modes','4'); c[1].metric('Completed runs',len(rows)); c[2].metric('Applied actions','5'); c[3].metric('Firefox targets','0')
 st.subheader('Experiment 001 at a glance')
 st.dataframe(pd.DataFrame([{'Mode':m.replace('_',' '),'p95 scheduler lag mean (ms)':round(x['metrics']['scheduler_lag_p95_ms']['mean'],3),'Completion mean (s)':round(x['metrics']['workload_completion_s']['mean'],3),'Applied nice actions':x['metrics']['nice_actions_applied']['mean']} for m,x in summary['per_mode'].items()]),hide_index=True,use_container_width=True)
 st.info('Safety boundary: all 20 workloads completed normally; no process kills, cgroup changes, or kernel scheduler changes occurred. The five real nice actions were scoped static nice +5 changes to the disposable workload.')
elif page=='Live Monitor':
 st.header('Live monitor - observation and suggestions only')
 st.warning("This page collects one snapshot and evaluates existing policy in memory. It does not invoke BGOpt's action executor, change nice values or cgroups, or start a controller loop.")
 mode=st.selectbox('Classification path',['Heuristic','Frozen ML v4'])
 if mode=='Frozen ML v4': st.caption('Uses the frozen BGOpt-native v4 model. It changes only the displayed classification.')
 if st.button('Collect one safe snapshot',type='primary'):
  try: st.session_state.snapshot=live_snapshot(mode)
  except Exception as exc: st.error(f'Snapshot unavailable: {exc}')
 snap=st.session_state.get('snapshot')
 if snap:
  h=snap['host']; c=st.columns(5); c[0].metric('Host CPU',f"{float(h['cpu_percent']):.1f}%"); c[1].metric('Load / CPU',f"{float(h['load_1_norm']):.2f}"); c[2].metric('Memory',f"{float(h['memory_percent']):.1f}%"); c[3].metric('Swap',f"{float(h['swap_percent']):.1f}%"); c[4].metric('Protected processes',snap['protected_count'])
  p=snap['pressure']; st.info(f"Pressure: {p.get('level','unknown')} (score {p.get('score',0):.2f}); reasons: {', '.join(p.get('reasons',[])) or 'none'}.")
  st.dataframe(pd.DataFrame(snap['rows']),hide_index=True,use_container_width=True,height=500)
  st.caption('Suggested decisions are policy outputs only. No process is targeted or modified from this dashboard.')
elif page=='Experiment 001':
 st.header('Experiment 001 - final 20-run evaluation')
 st.caption('Five repetitions each: Linux Default, Static Nice +5, BGOpt Heuristic, and BGOpt ML-assisted.')
 c=st.columns(4)
 for card,m in zip(c,['linux_default','static_nice_plus_5','bgopt_heuristic','bgopt_ml_assisted']):
  x=mode_metrics(summary,m); card.metric(m.replace('_',' ').title(),f"{x['scheduler_lag_p95_ms']['mean']:.3f} ms",f"completion {x['workload_completion_s']['mean']:.3f} s")
 for n in ['scheduler_lag_p95_by_mode.svg','workload_completion_by_mode.svg','foreground_background_tradeoff.svg','intervention_count_by_mode.svg','ml_prediction_distribution.svg']:
  p=ROOT/'reports/figures'/n
  if p.exists(): components.html(p.read_text(encoding='utf-8'),height=390,scrolling=False)
 st.subheader('All 20 raw run rows'); st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=420)
 st.warning('Caveat: ML-assisted had the lowest measured scheduler-lag values in this VM experiment, but controller iteration counts differ by mode. These are exploratory results, not broad production claims.')
elif page=='ML Analysis':
 st.header('ML analysis - provenance, boundaries, and observed labels')
 st.markdown('''- **LAPS** informed workload taxonomy and feature-design guidance; it was not used as direct runtime action labels.
- **BGOpt-native labelled psutil telemetry** is the deployable training data for the frozen runtime classifier.
- The classifier supports workload characterisation only; pressure, protection, and policy remain separate safeguards.''')
 st.subheader('Frozen classifier'); st.write('HistGradientBoostingClassifier with median imputation; 18 psutil-compatible runtime features; five classes: idle, interactive, background_cpu, memory_heavy, and io_heavy.')
 p=summary['per_mode']['bgopt_ml_assisted'].get('ml_predictions',{}); st.bar_chart(pd.DataFrame.from_dict(p,orient='index',columns=['telemetry records']))
 st.write('Aggregate observed records: background_cpu 10, interactive 40, io_heavy 685.')
 st.subheader('Independent workload-validation context'); st.write('The prior independent real-workload validation was predominantly interactive (71.1%) with background_cpu at 28.9%. This limitation remains explicit: the ML component is a constrained student-project characteriser, not a production classifier.')
 st.info('The dashboard does not retrain, alter features, or modify model artifacts.')

else:
 st.header('Run project')
 st.subheader('Safe live demonstration')
 st.write('This runs the existing Linux-default evaluation once with the disposable workload and returns a fresh measurement. It does not invoke BGOpt policy actions, change nice values, or modify cgroups.')
 repetition=st.number_input('Demo repetition',min_value=1,max_value=999,value=1,step=1)
 if st.button('Run safe Linux-default evaluation',type='primary'):
  from utils import run_safe_evaluation
  with st.spinner('Running the safe evaluation...'):
   try: st.session_state.safe_evaluation=run_safe_evaluation(int(repetition))
   except Exception as exc: st.error(f'Evaluation failed: {exc}')
 latest_result=Path('/tmp')/f'bgopt_dashboard_evaluation_{int(repetition)}.json'
 if not st.session_state.get('safe_evaluation') and latest_result.exists():
  import json
  st.session_state.safe_evaluation=json.loads(latest_result.read_text(encoding='utf-8'))
 if st.session_state.get('safe_evaluation'):
  st.success('Safe evaluation completed.')
  st.json(st.session_state.safe_evaluation)
 st.divider()
 st.subheader('Controlled Experiment 001')
 st.warning('A fresh full Experiment 001 includes the static nice +5 condition. That condition temporarily changes priority only for the disposable workload. It is not started from this dashboard by default.')
 acknowledged=st.checkbox('I understand that a full controlled experiment can make a real nice-priority change to the disposable workload.')
 st.button('Start controlled Experiment 001',disabled=True,help='Disabled: the current backend exposes safe single-mode evaluation, not a supported one-click full Experiment 001 runner.')
 st.caption('The disabled control prevents a dashboard from implying that an unsupported orchestration path is safe or available.')
