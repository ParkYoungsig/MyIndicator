import json
from pathlib import Path
from research.backtester.config import load_dual_engine_config

def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    else:
        out[prefix] = d
    return out

# 1) 현재 YAML 기반 effective config
cfg_yaml = load_dual_engine_config("backtester", logic_name=None, json_run=None)

# 2) JSON(run_dir) 기반 config_merged.json 로드
# backtester.yaml의 RESULTS.root가 ../results 이므로 프로젝트 기준 경로 맞춰주세요.
run_dir = Path("../results/backtester_20260129_142957")
cfg_json = json.loads((run_dir / "config_merged.json").read_text(encoding="utf-8"))

fy = flatten(cfg_yaml)
fj = flatten(cfg_json)

keys = sorted(set(fy) | set(fj))
diffs = []
for k in keys:
    if fy.get(k) != fj.get(k):
        diffs.append((k, fy.get(k), fj.get(k)))

print(f"DIFF COUNT = {len(diffs)}")
for k, a, b in diffs[:200]:
    print(f"- {k}\n  YAML: {a}\n  JSON: {b}\n")
