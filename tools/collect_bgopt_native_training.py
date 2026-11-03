"""Collect a bounded BGOpt-native runtime-feature training corpus."""
from __future__ import annotations
import csv, subprocess, sys, time
from pathlib import Path
from bgopt.features import MODEL_FEATURES
from bgopt.telemetry import TelemetryCollector
SPECS = {
    "idle": ("import time; time.sleep(2.2)", True),
    "interactive": ("import time; time.sleep(2.2)", False),
    "background_cpu": ("import hashlib; [hashlib.sha256(str(i).encode()).digest() for i in range(1800000)]", True),
    "memory_heavy": ("import time; x=bytearray(128*1024*1024); x[:]=b'x'*len(x); time.sleep(2.2)", True),
    "io_heavy": ("import os,time; p='/tmp/bgopt_native_io.bin'; f=open(p,'wb'); b=b'x'*(1024*1024); end=time.monotonic()+2.2;\nwhile time.monotonic()<end: f.seek(0); f.write(b); f.flush(); os.fsync(f.fileno());\nf.close(); os.unlink(p)", True),
}
def collect(output: str|Path, runs_per_class: int=8, interval: float=.20) -> int:
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
    collector=TelemetryCollector(current_user_only=True, exclude_process_names=("bgopt",))
    for label,(code,new_session) in SPECS.items():
        for run in range(1,runs_per_class+1):
            group=f"native_ml_{label}_{run:02d}"
            process=subprocess.Popen([sys.executable,"-c",code], start_new_session=new_session)
            samples=0
            while process.poll() is None and samples < 8:
                time.sleep(interval)
                for record in collector.collect():
                    if int(record.get("pid",-1)) != process.pid: continue
                    row={name: record.get(name,0.0) for name in MODEL_FEATURES}
                    row.update({"workload_label":label,"experiment_id":"bgopt_native_ml_001","scenario":label,"group":group,"pid":process.pid,"process_create_time":record.get("process_create_time")})
                    rows.append(row); samples += 1; break
            process.wait()
            if not samples: raise RuntimeError(f"no telemetry captured for {group}")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=[*MODEL_FEATURES,"workload_label","experiment_id","scenario","group","pid","process_create_time"]); writer.writeheader(); writer.writerows(rows)
    return len(rows)
if __name__ == "__main__":
    target=Path(sys.argv[1]) if len(sys.argv)>1 else Path("data/processed/bgopt_native_training.csv")
    print({"rows":collect(target),"output":str(target),"features":list(MODEL_FEATURES),"classes":list(SPECS)})
