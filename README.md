# 🌿 GreenGold — AI-Driven Datacenter Intelligence Platform

> **Reinforcement Learning for Sustainable GPU Workload Optimization**

GreenGold is an autonomous datacenter management system that uses a **Q-Learning / DQN Reinforcement Learning agent** to simultaneously minimize carbon emissions, energy costs, and thermal anomalies across a geographically distributed GPU datacenter — without pre-programmed rules or human intervention.

---

## ✨ Features

- 🤖 **RL Agent** — Q-Learning / Deep Q-Network (DQN) agent learns optimal workload distribution through trial and error
- 🌡️ **Physics-Based Simulation** — Temperature is causally linked to GPU utilization; agent actions produce realistic, observable effects
- 🌍 **Multi-Region Carbon Optimization** — Workloads are migrated from fossil-fuel regions (650 gCO₂/kWh) to renewable-energy regions (11 gCO₂/kWh)
- 📊 **Live Web Dashboard** — Real-time visualization of 80 GPU chips across 10 servers, with agent decision log
- 🔬 **Offline Analytics Pipeline** — Data generation → enrichment → RL-driven analysis → migration planning
- 📈 **Proven Results** — Agent reduces thermal anomalies from 35+ chips to 0 and stabilizes temperature from 82°C to 53°C

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Live Dashboard (Browser)                │
│   KPI Cards │ Live Charts │ Server Grid │ RL Agent Panel  │
│              ▲ polls every 2 seconds                      │
└──────────────┼───────────────────────────────────────────┘
               │ JSON REST API
┌──────────────┼───────────────────────────────────────────┐
│         Flask Server + RL Agent (app.py)                  │
│   REST API (5 endpoints) │ Q-Agent │ Physics Engine       │
└──────────────────────────────────────────────────────────┘

┌──────────────── Offline Pipeline ────────────────────────┐
│   data_generator → analytics_agent → migration_planner   │
└──────────────────────────────────────────────────────────┘
```

### Simulated Infrastructure

| Layer | Detail |
|---|---|
| Regions | 5 (Iceland, USA-Virginia, Denmark, India-Mumbai, France) |
| Servers | 10 (distributed across regions) |
| GPU Chips | 80 (8 per server — NVIDIA H100, A100, AMD MI300X) |
| Tick Rate | 1 decision every 2 seconds |

---

## 🤖 Reinforcement Learning

The agent is formulated as a **Markov Decision Process (MDP)**:

| MDP Component | Description |
|---|---|
| **State** | 48 discrete states — avg utilization (4 buckets) × avg temp (4 buckets) × avg carbon (3 buckets) |
| **Actions** | 4 — Hold, Shift Hottest Chip, Rebalance Utilization, Migrate High-Carbon |
| **Reward** | Multi-objective: utilization score + carbon penalty + temperature penalty + anomaly penalty |

### Agent Actions

| Action | Trigger Condition | Effect |
|---|---|---|
| **Hold** | System is stable | No change; rewards stability |
| **Shift Hottest Chip** | Any chip > 65°C | Migrates load to coolest server |
| **Rebalance Utilization** | Chips > 70% AND chips < 30% | Equalizes load across servers |
| **Migrate High-Carbon** | Carbon > 300 AND green server available | Moves compute to low-carbon region |

### Learning Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.15 |
| Discount factor (γ) | 0.95 |
| Initial exploration (ε) | 0.40 |
| ε decay | 0.997/episode → min 0.05 |

---

## 📊 Results

| Metric | Episode 0 | Episode 1300+ |
|---|---|---|
| Avg Temperature | ~82°C | ~53°C |
| Anomaly Count | 35+ chips | 0 chips |
| Avg Reward | Variable | Stable at +0.63 |
| Dominant Action | Random | Hold (system stabilized) |

---

## 📁 Project Structure

```
ECO_GOLD-main/
├── app.py                # Flask server + Q-Learning agent + simulation engine
├── index.html            # Live dashboard (vanilla HTML/CSS/JS + Chart.js)
├── datacenter_env.py     # Gym-style DatacenterEnv with physics-based simulation
├── rl_agent.py           # DQN agent (PyTorch) — train, act, save/load
├── main.py               # Offline pipeline orchestrator (run end-to-end)
├── data_generator.py     # Synthetic telemetry & infrastructure CSV generator
├── analytics_agent.py    # Offline analytics: enrichment, forecasting, visualization
├── migration_planner.py  # Self-healing migration with financial audit report
├── gpu_monitor.py        # Real NVIDIA GPU telemetry via pynvml (optional)
├── workload_runner.py    # Workload execution and scheduling utilities
├── requirements.txt      # Python dependencies
└── Procfile              # Gunicorn deployment config
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/ECO_GOLD.git
cd ECO_GOLD
pip install -r requirements.txt
```

### Run the Live Dashboard

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.  
The RL agent starts learning immediately — watch it stabilize the datacenter in real time.

### Run the Offline Analytics Pipeline

```bash
python main.py
```

This will:
1. Generate synthetic telemetry data
2. Train the DQN agent for 2000 steps
3. Run RL-driven analytics and produce charts
4. Generate a `migration_report.csv` with financial audit

---

## 🌍 Region Energy Profiles

| Region | Energy Source | Carbon (gCO₂/kWh) | Cost ($/kWh) | Ambient Temp |
|---|---|---|---|---|
| 🇮🇸 Iceland | Geothermal/Hydro | 20 | $0.05 | 5°C |
| 🇩🇰 Denmark | Wind | 11 | $0.28 | 9°C |
| 🇫🇷 France | Nuclear | 12 | $0.18 | 14°C |
| 🇺🇸 USA (Virginia) | Coal/Gas Mix | 650 | $0.12 | 18°C |
| 🇮🇳 India (Mumbai) | Solar/Coal Mix | 450 | $0.09 | 28°C |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.0 |
| RL Framework | PyTorch (DQN), NumPy (Q-table) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js 4.4 |
| Data Analysis | Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Production Server | Gunicorn (via Procfile) |

---

## 🔮 Future Work

- **Deep RL at scale** — Replace tabular Q-learning with PPO/SAC for thousands of servers
- **Multi-Agent RL** — One agent per region, coordinating via communication channels
- **Live Carbon API** — Integrate with [electricityMap](https://www.electricitymap.org/) or [WattTime](https://watttime.org/) for real-time carbon intensity
- **Kubernetes Operator** — Deploy as a K8s controller managing real pod scheduling
- **Predictive Workloads** — LSTM/Transformer to forecast incoming demand

---

## 📄 License

This project is for educational and research purposes.

---

<div align="center">
  <strong>Built with ⚡ by the GreenGold Team</strong><br/>
  <em>Making AI infrastructure greener, one decision at a time.</em>
</div>
