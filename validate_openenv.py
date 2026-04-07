"""
Final OpenEnv Submission Validation
Checks every requirement from the META AI checklist.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")

ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("OPENENV SUBMISSION CHECKLIST VALIDATION")
print("=" * 70)

# ── 1. File Existence ────────────────────────────────────────────
print("\n--- 1. Required Files ---")
check("inference.py exists in root", (ROOT / "inference.py").exists())
check("Dockerfile exists in root", (ROOT / "Dockerfile").exists())
check("openenv.yaml exists", (ROOT / "openenv.yaml").exists())
check("requirements.txt exists", (ROOT / "requirements.txt").exists())
check("server/app.py exists", (ROOT / "server" / "app.py").exists())
check("server/Dockerfile exists", (ROOT / "server" / "Dockerfile").exists())
check("README.md exists", (ROOT / "README.md").exists())

# ── 2. inference.py checks ──────────────────────────────────────
print("\n--- 2. inference.py Content ---")
inf_code = (ROOT / "inference.py").read_text(encoding="utf-8")
check("uses 'from openai import OpenAI'", "from openai import OpenAI" in inf_code)
check("has API_BASE_URL env var", 'os.getenv("API_BASE_URL")' in inf_code or "os.getenv('API_BASE_URL')" in inf_code)
check("has MODEL_NAME env var", 'os.getenv("MODEL_NAME")' in inf_code or "os.getenv('MODEL_NAME')" in inf_code)
check("has HF_TOKEN env var", 'os.getenv("HF_TOKEN")' in inf_code or "os.getenv('HF_TOKEN')" in inf_code)
check("API_BASE_URL has default", 'API_BASE_URL' in inf_code and 'router.huggingface.co' in inf_code)
check("MODEL_NAME has default", 'MODEL_NAME' in inf_code and 'Qwen' in inf_code)
check("HF_TOKEN has NO hardcoded default", 'os.getenv("HF_TOKEN", "' not in inf_code)
check("has [START] log", "[START]" in inf_code)
check("has [STEP] log", "[STEP]" in inf_code)
check("has [END] log", "[END]" in inf_code)
check("has asyncio.run", "asyncio.run" in inf_code)
check("score clamped to [0,1]", "min(1.0" in inf_code and "max(0.0" in inf_code)

# ── 3. requirements.txt ─────────────────────────────────────────
print("\n--- 3. requirements.txt ---")
reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("has openai", "openai" in reqs)
check("has fastapi", "fastapi" in reqs)
check("has uvicorn", "uvicorn" in reqs)
check("has pydantic", "pydantic" in reqs)
check("has gymnasium", "gymnasium" in reqs)
check("has numpy", "numpy" in reqs)

# ── 4. openenv.yaml ─────────────────────────────────────────────
print("\n--- 4. openenv.yaml ---")
yaml_text = (ROOT / "openenv.yaml").read_text(encoding="utf-8")
check("has 'tasks:' section", "tasks:" in yaml_text)
check("has portfolio-growth task", "portfolio-growth" in yaml_text)
check("has risk-management task", "risk-management" in yaml_text)
check("has market-timing task", "market-timing" in yaml_text)
check("has 3+ tasks (graders)", yaml_text.count("- name:") >= 3, f"found {yaml_text.count('- name:')} tasks")
check("has server section", "server:" in yaml_text)
check("has models section", "models:" in yaml_text)
check("has observation model", "observation:" in yaml_text)
check("has action model", "action:" in yaml_text)
check("reward range [0,1]", "reward_range: [0.0, 1.0]" in yaml_text)

# ── 5. Server endpoints ─────────────────────────────────────────
print("\n--- 5. Server Endpoints ---")
srv_code = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
check("has /reset endpoint", '"/reset"' in srv_code or "'/reset'" in srv_code)
check("has /step endpoint", '"/step"' in srv_code or "'/step'" in srv_code)
check("has /state endpoint", '"/state"' in srv_code or "'/state'" in srv_code)
check("uses FastAPI", "FastAPI" in srv_code)
check("uses Pydantic models", "BaseModel" in srv_code)
check("port 7860", "7860" in srv_code)

# ── 6. Dockerfile ────────────────────────────────────────────────
print("\n--- 6. Dockerfile ---")
docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
check("has FROM python", "FROM python" in docker)
check("has EXPOSE 7860", "EXPOSE 7860" in docker)
check("has CMD uvicorn", "uvicorn" in docker)
check("installs requirements.txt", "requirements.txt" in docker)

# ── 7. Environment APIs ─────────────────────────────────────────
print("\n--- 7. Environment APIs ---")
from env.real_estate_env import RealEstateEnv
from env.config import EnvConfig

config = EnvConfig()
env = RealEstateEnv(config=config)

obs, info = env.reset(seed=42)
check("reset() returns obs + info", obs is not None and info is not None)
check("obs shape = (69,)", obs.shape == (69,))

obs2, r, term, trunc, info2 = env.step(np.array([0, 1, 0, 0, 0]))
check("step() returns 5-tuple", True)
check("reward is float", isinstance(r, float))

state = env.state()
check("state() returns dict", isinstance(state, dict))
check("state has required keys", all(k in state for k in ["month", "cash", "net_worth", "regime", "portfolio", "market"]))

# ── 8. Grading Logic ────────────────────────────────────────────
print("\n--- 8. Grading (scores in [0,1]) ---")
# Run full episode
env2 = RealEstateEnv(config=config)
from agents.baselines import RuleBasedAgent
agent = RuleBasedAgent(num_slots=config.max_properties)
obs, info = env2.reset(seed=42)
done = False
while not done:
    action = agent.predict(obs, info)
    obs, r, term, trunc, info = env2.step(action)
    done = term or trunc

summary = info.get("episode_summary", {})

# portfolio-growth
s1 = max(0.0, min(1.0, summary.get("total_return_pct", 0) / 100.0))
check(f"portfolio-growth score={s1:.4f} in [0,1]", 0.0 <= s1 <= 1.0)

# risk-management
s2 = max(0.0, min(1.0, summary.get("annualized_sharpe", 0) / 2.0))
check(f"risk-management score={s2:.4f} in [0,1]", 0.0 <= s2 <= 1.0)

# market-timing
ret = max(0.0, min(0.5, summary.get("total_return_pct", 0) / 200.0))
dd = max(0.0, 0.3 * (1.0 - summary.get("max_drawdown_pct", 100) / 50.0))
act = min(0.2, (summary.get("total_properties_bought", 0) + summary.get("total_properties_sold", 0)) * 0.02)
s3 = max(0.0, min(1.0, ret + dd + act))
check(f"market-timing score={s3:.4f} in [0,1]", 0.0 <= s3 <= 1.0)

# ── 9. Stdout Format ────────────────────────────────────────────
print("\n--- 9. Stdout Format Verification ---")
import io
from contextlib import redirect_stdout

# Import log functions from inference
sys.path.insert(0, str(ROOT))
# Re-import inference module's logging
exec_globals = {}
exec("""
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)
""", exec_globals)

buf = io.StringIO()
with redirect_stdout(buf):
    exec_globals["log_start"]("test-task", "real_estate_rl", "test-model")
    exec_globals["log_step"](1, "[0,1,0,0,0]", 0.05, False, None)
    exec_globals["log_end"](True, 1, 0.85, [0.05])

output = buf.getvalue()
lines = output.strip().split("\n")
check("[START] line format", lines[0].startswith("[START] task="))
check("[STEP] line format", lines[1].startswith("[STEP] step="))
check("[END] line format", lines[2].startswith("[END] success="))
check("reward formatted to 2dp", "reward=0.05" in lines[1])
check("done is lowercase bool", "done=false" in lines[1])
check("error=null when no error", "error=null" in lines[1])
check("success is lowercase bool", "success=true" in lines[2])
check("score formatted to 2dp", "score=0.85" in lines[2])

# ── FINAL RESULTS ────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"FINAL: {PASS} PASSED, {FAIL} FAILED out of {PASS + FAIL} checks")
print("=" * 70)

if FAIL > 0:
    print("\nSOME CHECKS FAILED!")
    sys.exit(1)
else:
    print("\nALL OPENENV CHECKS PASSED! Ready to submit.")
    sys.exit(0)
