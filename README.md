# 🌿 GreenGold — AI-Driven Datacenter Intelligence

> A Deep Q-Network (DQN) reinforcement learning agent that learns to manage GPU workloads across a globally distributed datacenter — optimizing for temperature, carbon emissions, and energy efficiency in real time.

---

## What Is This?

Modern GPU datacenters face a constant three-way trade-off: keep chips from overheating, minimize carbon emissions, and balance energy costs. Traditional systems rely on fixed rules like "migrate when temperature exceeds 85°C." These rules can't adapt, and they optimize for only one thing at a time.

GreenGold replaces those rules with a **DQN reinforcement learning agent** that learns from experience. It observes the live state of 10 servers across 5 global regions, picks an action every second, and gradually learns which decisions lead to a healthier, greener datacenter. No hard-coded thresholds. No human intervention.

A live web dashboard shows everything in real time — chip temperatures, carbon footprints, agent decisions, reward history, and anomaly alerts.

---

## Live Demo Preview

```
10 servers  ×  8 GPU chips  =  80 chips total
5 regions: Iceland · Denmark · France · USA (Virginia) · India (Mumbai)
Agent ticks every 1 second, dashboard refreshes every 2 seconds
```

The agent starts exploring randomly (ε = 40%) and gradually shifts to exploitation (ε = 5%) as it learns what works. You can watch this happen live on the dashboard.

---

## Project Structure

```
Green-Gold/
│
├── app.py                # Flask server + background RL tick thread
├── datacenter_env.py     # Gym-style datacenter environment (physics engine)
├── rl_agent.py           # DQN agent (PyTorch + pure-numpy fallback)
├── analytics_agent.py    # Offline analytics — runs RL on historical telemetry
├── migration_planner.py  # RL-driven migration report generator
├── data_generator.py     # Generates infrastructure registry + telemetry CSVs
├── main.py               # Full pipeline runner (generate → train → analyze)
├── gpu_monitor.py        # Real GPU telemetry via NVML (optional)
├── workload_runner.py    # Real OS worker processes (optional)
├── index.html            # Dashboard frontend (Chart.js, no build step needed)
├── Procfile              # Gunicorn config for deployment
└── requirements.txt      # Python dependencies
```

---

## How It Works

### The Environment

`datacenter_env.py` simulates a datacenter with **physics-based temperature modeling**:

```
target_temp  =  ambient + 15 + (utilization × 0.65) + noise(±1.5°C)
actual_temp  =  0.6 × previous_temp + 0.4 × target_temp
```

This means temperature is causally linked to workload — reduce load on a chip and it visibly cools down over the next few ticks. The five regions each have different ambient temperatures and carbon profiles:

| Region | Energy Source | Carbon (gCO₂/kWh) | Cost ($/kWh) | Ambient |
|---|---|---|---|---|
| Iceland | Geothermal/Hydro | 20 | $0.05 | 5°C |
| Denmark | Wind | 11 | $0.28 | 9°C |
| France | Nuclear | 12 | $0.18 | 14°C |
| USA (Virginia) | Coal/Gas Mix | 650 | $0.12 | 18°C |
| India (Mumbai) | Solar/Coal Mix | 450 | $0.09 | 28°C |

### The Agent

`rl_agent.py` implements a **Deep Q-Network** with:

- **Policy network**: MLP with architecture `26 → 128 → 64 → 7`
- **Target network**: Hard-copied from policy every 100 steps for stable training
- **Experience replay**: 50,000-transition buffer, random mini-batch sampling
- **Epsilon-greedy**: Linear decay from 1.0 → 0.05 over training
- **Fallback**: If PyTorch isn't installed, a pure-numpy DQN kicks in automatically

The agent observes a **26-dimensional state vector** including average utilization, max/mean temperatures, carbon intensity, anomaly counts, idle chip ratios, and per-server utilization breakdowns.

### The 7 Actions

| # | Action | What It Does |
|---|---|---|
| 0 | **Hold** | Do nothing — rewarded when system is stable |
| 1 | **Shift Hottest Chip** | Move 50% load from the hottest chip to the coolest server |
| 2 | **Rebalance Utilization** | Equalize load across up to 4 overloaded/underloaded chip pairs |
| 3 | **Migrate High-Carbon** | Move workload from dirty regions (>300 gCO₂) to clean ones (<50 gCO₂) |
| 4 | **Emergency Cool-Down** | Aggressively throttle all chips above 78°C |
| 5 | **Consolidate Idle** | Pack scattered idle workloads onto fewer servers |
| 6 | **Boost Green Utilization** | Shift work TO renewable-energy servers to maximize green usage |

### The Reward Function

Every tick the agent receives a reward signal that balances four objectives:

```
reward = action_reward
       + util_score          (penalty for being far from 55% target utilization)
       - carbon_penalty      (penalty proportional to average carbon intensity)
       - temp_penalty        (penalty when average temp exceeds 70°C)
       - anomaly_penalty     (−0.15 per anomalous chip)
       - action_cost         (−0.05 for any non-Hold action, to prefer minimal intervention)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

PyTorch is optional but recommended. The agent works without it using the built-in numpy fallback.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ManasTarare/Green-Gold.git
cd Green-Gold

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline (Recommended First Run)

This generates data, trains the agent for 2000 steps, runs analytics, and saves a dashboard image:

```bash
python main.py
```

You'll see output like:
```
===== STEP 1: DATA GENERATION =====
✅ Saved 'infrastructure_registry.csv' with 10 servers.
✅ Saved 'telemetry_log.csv' with 4,000 snapshots.

===== STEP 2: DQN AGENT TRAINING =====
  Step  200/2000 | ε=0.900 | avg_reward=0.312 | avg_loss=0.04521
  Step  400/2000 | ε=0.800 | avg_reward=0.489 | avg_loss=0.02187
  ...
  Step 2000/2000 | ε=0.050 | avg_reward=0.681 | avg_loss=0.00823
✅ Model saved as 'model.pt'
```

### Launch the Live Dashboard

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. The agent starts ticking immediately — you'll see chip temperatures, utilization, and agent decisions updating in real time.

---

## AI Assistant (Built-in)

The dashboard includes a chat panel powered by Claude. To enable it, set your Anthropic API key as an environment variable **before** starting the server:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-...
python app.py

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

The API key is read server-side and never exposed to the browser.

---

## Offline Analytics

### Run RL Analysis on Historical Data

```bash
python analytics_agent.py
```

This loads the telemetry CSVs, runs each historical timestep through the trained DQN, and generates `dashboard.png` — a 4-panel chart showing carbon footprint by region, chip temperature distributions, agent action recommendations, and anomaly breakdown.

### Generate a Migration Report

```bash
python migration_planner.py
```

This identifies anomalous chips using Z-score detection, queries the DQN agent for the optimal action for each one, and outputs `migration_report.csv` with recommended targets, estimated costs, and agent confidence scores.

---

## Real Hardware Support (Optional)

GreenGold runs in simulation mode by default, but can use real hardware if available.

**Real GPU telemetry** — if you have an NVIDIA GPU and `nvidia-ml-py` installed:
```bash
pip install nvidia-ml-py
```
The app auto-detects your GPU(s) and overlays real temperature and utilization data onto the simulation.

**Real OS worker processes** — `workload_runner.py` can spawn actual Python subprocesses doing numpy matrix work, and the agent's actions (pause, migrate, rebalance) translate to real SIGSTOP/SIGCONT signals and CPU affinity changes via `psutil`.

---

## Deploying to the Cloud

The project includes a `Procfile` for one-command deployment to platforms like Heroku, Render, or Railway.

```bash
# Example: Heroku
heroku create your-app-name
heroku config:set ANTHROPIC_API_KEY=sk-ant-...
git push heroku main
```

The Procfile uses `--workers 1 --threads 4` intentionally — the RL agent runs in a background thread and must share state within a single process.

---

## API Reference

The Flask server exposes these endpoints:

| Endpoint | Description |
|---|---|
| `GET /` | Serves the dashboard HTML |
| `GET /api/summary` | Live KPIs: avg utilization, temperature, anomaly count, epsilon |
| `GET /api/metrics` | Last 60 time-series data points (util, temp, reward, loss) |
| `GET /api/rl_log` | Last 50 agent decisions with action name, message, and reward |
| `GET /api/agent_stats` | Full agent statistics: episodes, rewards, loss history, action counts |
| `GET /api/servers` | Complete state of all 10 servers and 80 chips |
| `GET /api/workers` | Real OS worker process status (empty if not running) |
| `GET /api/save_model` | Saves current agent weights to `model.pt` or `model.npz` |
| `POST /api/chat` | AI assistant proxy (requires `ANTHROPIC_API_KEY` env var) |

---

## Key Design Decisions

**Why DQN instead of tabular Q-learning?**
A tabular Q-table can't generalize — it memorizes exact state-action pairs. The DQN's neural network generalizes across similar states, making it far more effective with a 26-dimensional continuous observation space.

**Why a background thread for the agent tick?**
Earlier versions triggered an agent step on every `/api/summary` HTTP request. This meant the agent only learned when the browser was open, and concurrent requests could stall. The background thread decouples learning from API traffic.

**Why `--workers 1` in Gunicorn?**
The RL agent's state lives in Python memory. With multiple Gunicorn workers, each process would have its own agent — they'd never share experience and the dashboard would show inconsistent data. Single worker + multiple threads is the right trade-off here.

---

## What the Agent Learns Over Time

| Phase | Epsilon | Behavior |
|---|---|---|
| Early (steps 0–500) | ~1.0 → 0.75 | Mostly random — exploring all actions |
| Mid (steps 500–1500) | 0.75 → 0.20 | Learning which actions reduce anomalies |
| Late (steps 1500+) | 0.20 → 0.05 | Consistent policy — holds when stable, acts decisively on anomalies |

A fully trained agent will typically hold when everything is fine, shift or cool-down when temperatures spike, and migrate toward green servers when high-carbon regions are heavily loaded.

---

## Known Limitations

- The thermal model is simplified — real datacenter cooling involves airflow, liquid cooling, and HVAC dynamics that aren't modeled
- Cross-region migration latency and data transfer costs aren't simulated
- Carbon intensity is static per region — in reality it fluctuates throughout the day
- The reward function uses fixed weights; different operators may want different priorities

---

## Future Ideas

- Integrate with [electricityMap](https://www.electricitymap.org) or [WattTime](https://www.watttime.org) for live carbon intensity data
- Replace the MLP with a Transformer-based policy for better long-range planning
- Add multi-agent RL — one agent per region that communicates with others
- Deploy as a Kubernetes operator that manages real pod scheduling
- Add LSTM-based workload forecasting so the agent can anticipate demand spikes

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

**Manas Tarare** — [github.com/ManasTarare](https://github.com/ManasTarare)

Built as an exploration of applying deep reinforcement learning to real-world infrastructure optimization problems.
