"""
app.py — GreenGold Flask server (fully upgraded).

Changes from old version:
  - Uses DatacenterEnv (datacenter_env.py) instead of inline simulation
  - Uses DQNAgent (rl_agent.py) instead of flat Q-table
  - Supports 7 actions (was 4)
  - Integrates GPUMonitor for real GPU telemetry when available
  - Integrates WorkloadRunner for real OS process management when available
  - Loads pre-trained model (model.pt / model.npz) on startup if it exists
  - Thread-safe: agent tick runs inside a lock so concurrent API calls don't corrupt state
"""

import os
import threading
from datetime import datetime
from flask import Flask, jsonify, send_file

# ── Core RL modules ──────────────────────────────────────────────
from datacenter_env import DatacenterEnv, OBS_DIM, N_ACTIONS, ACTION_NAMES
from rl_agent import DQNAgent, HAS_TORCH

# ── Optional real-hardware modules ──────────────────────────────
try:
    from gpu_monitor import GPUMonitor
    _gpu_monitor = GPUMonitor()
except Exception:
    _gpu_monitor = None

try:
    from workload_runner import WorkloadRunner
    _workload_runner = WorkloadRunner(num_workers=6)
    _workload_runner.start()
except Exception:
    _workload_runner = None

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════
# AGENT + ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════
env = DatacenterEnv(seed=42)
agent = DQNAgent(
    obs_dim=OBS_DIM,
    n_actions=N_ACTIONS,
    lr=1e-3,
    gamma=0.95,
    batch_size=64,
    epsilon_start=0.40,
    epsilon_end=0.05,
    epsilon_decay_steps=3000,
)

# Load pre-trained model if available
_MODEL_LOADED = False
for _model_path in ("model.pt", "model.npz"):
    if os.path.exists(_model_path):
        try:
            agent.load(_model_path)
            _MODEL_LOADED = True
            print(f"✅ Loaded pre-trained model from '{_model_path}'.")
            print(f"   ε = {agent.epsilon:.4f}  |  steps = {agent.steps_done}")
        except Exception as e:
            print(f"⚠️  Could not load '{_model_path}': {e}")
        break

if not _MODEL_LOADED:
    print("ℹ️  No pre-trained model found — agent starts fresh (ε=0.40).")

# Real GPU mode flag
_REAL_GPU = _gpu_monitor is not None and _gpu_monitor.available
_REAL_WORKERS = _workload_runner is not None and _workload_runner.available
if _REAL_GPU:
    print(f"✅ Real GPU telemetry active — {_gpu_monitor.gpu_count} GPU(s) detected.")
else:
    print("ℹ️  No NVIDIA GPU found — running in simulation mode.")
if _REAL_WORKERS:
    print("✅ Real OS worker processes active.")

# ══════════════════════════════════════════════════════════════════
# SHARED STATE (protected by lock)
# ══════════════════════════════════════════════════════════════════
_lock          = threading.Lock()
_obs           = env.reset()
_metrics_log   = []   # last 60 time-series points
_rl_log        = []   # last 50 agent decisions
_TICK_INTERVAL = 1.0  # seconds between background agent ticks


# ══════════════════════════════════════════════════════════════════
# AGENT TICK
# ══════════════════════════════════════════════════════════════════
def _agent_tick():
    """
    One full RL step:
      1. Optionally overlay real GPU data onto the env state
      2. Get observation from env
      3. Agent picks action (ε-greedy)
      4. Env executes action, returns reward + next obs
      5. Agent stores transition and runs one gradient update
      6. Epsilon decays
      7. Logs are updated
    """
    global _obs

    with _lock:
        # ── (Optional) Real GPU overlay ──────────────────────────
        if _REAL_GPU:
            real_gpus = _gpu_monitor.read_all()
            for idx, gpu in enumerate(real_gpus):
                if idx >= len(env.servers):
                    break
                srv = env.servers[idx]
                chip = srv["chips"][0]
                chip["utilization"] = float(gpu["util_pct"])
                chip["temp"]        = float(gpu["temp_c"])
                chip["vram_used"]   = round(gpu["vram_used_mb"] / 1024, 1)
                chip["anomaly"]     = chip["temp"] > 82 or chip["utilization"] > 92

        # ── Agent step ───────────────────────────────────────────
        action_idx              = agent.select_action(_obs)
        next_obs, reward, done, info = env.step(action_idx)
        agent.store_transition(_obs, action_idx, reward, next_obs, done)
        agent.train_step()
        agent.decay_epsilon()
        _obs = next_obs

        if done:
            agent.end_episode(env.episode_reward)
            _obs = env.reset()

        # ── (Optional) Real worker action ────────────────────────
        if _REAL_WORKERS:
            if action_idx == 1:
                hw = _workload_runner.get_hottest_worker()
                if hw:
                    _workload_runner.pause_worker(hw.worker_id)
            elif action_idx == 2:
                _workload_runner.rebalance()
            elif action_idx == 3:
                _workload_runner.migrate_worker(0, "SRV-WRK-B")
            elif action_idx == 4:
                for w in _workload_runner.get_all():
                    if not w["paused"]:
                        _workload_runner.pause_worker(w["worker_id"])

        # ── Logging ──────────────────────────────────────────────
        now   = datetime.now().strftime("%H:%M:%S")
        stats = agent.get_stats()
        summary = env.get_summary()

        _metrics_log.append({
            "time":      now,
            "avg_util":  summary["avg_util"],
            "avg_temp":  summary["avg_temp"],
            "anomalies": summary["anomaly_count"],
            "reward":    reward,
            "loss":      round(stats.get("recent_loss", 0.0), 6),
        })
        if len(_metrics_log) > 60:
            _metrics_log.pop(0)

        _rl_log.insert(0, {
            "time":   now,
            "action": ACTION_NAMES[action_idx],
            "msg":    info.get("message", ""),
            "reward": reward,
        })
        if len(_rl_log) > 50:
            _rl_log.pop()

# ══════════════════════════════════════════════════════════════════
# BACKGROUND TICK THREAD
# ══════════════════════════════════════════════════════════════════
def _tick_loop():
    """Continuously tick the agent at a fixed interval in a background thread."""
    while True:
        try:
            _agent_tick()
        except Exception as e:
            print(f"⚠️  Tick error: {e}")
        time.sleep(_TICK_INTERVAL)

import time as _time_module
import time

_ticker = threading.Thread(target=_tick_loop, daemon=True)
_ticker.start()


@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return send_file(html_path, mimetype="text/html")


@app.route("/api/summary")
def api_summary():
    """Return live KPI metrics (agent ticks in background thread)."""
    with _lock:
        summary = env.get_summary()
        stats   = agent.get_stats()
    return jsonify({
        "avg_util":      summary["avg_util"],
        "avg_temp":      summary["avg_temp"],
        "anomaly_count": summary["anomaly_count"],
        "total_chips":   summary["total_chips"],
        "greenest_loc":  summary["greenest_loc"],
        "epsilon":       stats["epsilon"],
        "real_gpu":      _REAL_GPU,
        "real_workers":  _REAL_WORKERS,
        "model_loaded":  _MODEL_LOADED,
    })


@app.route("/api/metrics")
def api_metrics():
    with _lock:
        return jsonify(list(_metrics_log))


@app.route("/api/rl_log")
def api_rl_log():
    with _lock:
        return jsonify(list(_rl_log))


@app.route("/api/agent_stats")
def api_agent_stats():
    with _lock:
        stats = agent.get_stats()
    return jsonify({
        "episode":       stats["episodes"],
        "steps_done":    stats["steps_done"],
        "train_steps":   stats["train_steps"],
        "avg_reward":    stats["avg_reward"],
        "total_reward":  stats["total_reward"],
        "actions_taken": stats["steps_done"],
        "buffer_size":   stats["buffer_size"],
        "recent_loss":   stats.get("recent_loss", 0.0),
        "network_arch":  stats.get("network_arch", "DQN"),
        "action_counts": stats.get("action_counts", []),
        "reward_history": stats.get("reward_history", []),
        "loss_history":  stats.get("loss_history", []),
    })


@app.route("/api/servers")
def api_servers():
    with _lock:
        servers = env.get_server_data()
        real_gpu_data = {}
        if _REAL_GPU:
            for gpu in _gpu_monitor.read_all():
                real_gpu_data[gpu["gpu_index"]] = gpu

        result = []
        for i, srv in enumerate(servers):
            chips_out = []
            for j, c in enumerate(srv["chips"]):
                chip_entry = {
                    "chip_id":     c["chip_id"],
                    "utilization": c["utilization"],
                    "base_load":   c["base_load"],
                    "temp":        c["temp"],
                    "anomaly":     c["anomaly"],
                    "task":        c["task"],
                    "cooldown":    c["cooldown"],
                    "source":      "sim",
                }
                if j == 0 and i in real_gpu_data:
                    g = real_gpu_data[i]
                    chip_entry["utilization"] = g["util_pct"]
                    chip_entry["temp"]        = g["temp_c"]
                    chip_entry["source"]      = "real"
                    chip_entry["power_w"]     = g.get("power_w")
                    chip_entry["gpu_name"]    = g.get("name", "")
                chips_out.append(chip_entry)

            result.append({
                "server_id": srv["server_id"],
                "location":  srv["location"],
                "ambient":   srv["ambient"],
                "carbon":    srv["carbon"],
                "chips":     chips_out,
            })
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Proxy AI assistant requests to Anthropic — keeps API key server-side."""
    import requests as _req
    from flask import request as flask_request
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured on server"}), 500
    body = flask_request.get_json(silent=True) or {}
    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": body.get("system", "You are a helpful datacenter assistant."),
                "messages": body.get("messages", []),
            },
            timeout=30,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/workers")
def api_workers():
    """Return real OS worker process status (empty list if not active)."""
    if _REAL_WORKERS:
        return jsonify(_workload_runner.get_all())
    return jsonify([])


@app.route("/api/save_model")
def api_save_model():
    """Save the current agent weights to disk."""
    # FIX: removed dead `if True else "model.npz"` — pick format based on HAS_TORCH
    primary_path = "model.pt" if HAS_TORCH else "model.npz"
    fallback_path = "model.npz"
    try:
        agent.save(primary_path)
        stats = agent.get_stats()
        return jsonify({
            "ok": True,
            "path": primary_path,
            "steps": stats["steps_done"],
            "epsilon": stats["epsilon"],
        })
    except Exception as e:
        try:
            agent.save(fallback_path)
            return jsonify({"ok": True, "path": fallback_path})
        except Exception as e2:
            return jsonify({"ok": False, "error": str(e2)}), 500


# ══════════════════════════════════════════════════════════════════
# SHUTDOWN HOOK
# ══════════════════════════════════════════════════════════════════
import atexit

def _on_exit():
    try:
        save_path = "model.pt" if HAS_TORCH else "model.npz"
        agent.save(save_path)
        print("✅ Model auto-saved on exit.")
    except Exception:
        try:
            agent.save("model.npz")
        except Exception:
            pass
    if _REAL_WORKERS:
        _workload_runner.shutdown()
    if _REAL_GPU and _gpu_monitor:
        _gpu_monitor.shutdown()

atexit.register(_on_exit)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("   GreenGold — DQN Datacenter Intelligence")
    print(f"   Actions : {N_ACTIONS}  |  Obs dim : {OBS_DIM}")
    print(f"   Real GPU: {_REAL_GPU}  |  Real workers: {_REAL_WORKERS}")
    print(f"   Port    : {port}")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=port)
