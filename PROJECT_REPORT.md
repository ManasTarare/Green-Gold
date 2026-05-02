# GreenGold: AI-Driven Datacenter Intelligence Platform
## A Reinforcement Learning Approach to Sustainable GPU Workload Optimization

---

# Chapter 1: Introduction

## 1.1 Background

The rapid growth of artificial intelligence and cloud computing has led to an unprecedented expansion of global datacenter infrastructure. Modern GPU datacenters — powering large language models, image generation, scientific simulations, and vector search — consume enormous amounts of electricity and generate significant carbon emissions. The International Energy Agency estimates that datacenters account for approximately 1-1.5% of global electricity consumption, a figure projected to grow as AI workloads intensify.

Datacenter operators face a fundamental trilemma when managing GPU workloads:

1. **Carbon Emissions**: Regions powered by fossil fuels (coal, natural gas) offer low-cost compute but produce high carbon emissions (up to 650 gCO2/kWh). Green energy regions (geothermal, wind, hydro) produce minimal emissions but may be more expensive or capacity-constrained.

2. **Energy Cost**: Electricity pricing varies dramatically across regions — from $0.05/kWh in Iceland (geothermal) to $0.28/kWh in Denmark (wind). Operators must balance sustainability goals against operational budgets.

3. **Thermal Efficiency**: GPU chips generate substantial heat under load. Overloaded chips throttle performance, risk hardware damage, and trigger anomalies. Ambient temperature varies by region (5°C in Iceland vs 28°C in Mumbai), directly affecting cooling requirements.

Traditional approaches to workload management rely on static rules (e.g., "migrate when temperature exceeds threshold X") or manual operator intervention. These approaches fail to capture the complex, multi-objective nature of the optimization problem and cannot adapt to changing conditions in real time.

## 1.2 Problem Statement

Design and implement an autonomous system that can learn to optimally distribute GPU workloads across a geographically distributed datacenter network, simultaneously minimizing carbon emissions, energy costs, and thermal anomalies — without requiring pre-programmed rules or human intervention.

## 1.3 Objectives

1. Build a physics-based simulation of a multi-region GPU datacenter with realistic thermal dynamics
2. Implement a Reinforcement Learning agent (Q-Learning) that learns workload optimization through trial and error
3. Create a real-time web dashboard for monitoring agent decisions and datacenter health
4. Develop offline analytics tools for historical telemetry analysis and migration planning
5. Demonstrate that the RL agent can reduce anomalies, stabilize temperatures, and shift workloads to greener regions

## 1.4 Scope

The system simulates 10 servers across 5 global regions, each containing 8 GPU chips (80 chips total). The RL agent operates in real-time, making one decision every 2 seconds. The simulation includes physics-based temperature modeling, workload transfer mechanics, and anomaly detection. A live web dashboard provides visualization of all system metrics.

---

# Chapter 2: Literature Review

## 2.1 Reinforcement Learning in Resource Management

Reinforcement Learning (RL) has emerged as a powerful paradigm for sequential decision-making in complex environments. Unlike supervised learning, RL agents learn optimal policies through interaction with an environment, receiving reward signals that guide behavior.

**Q-Learning** (Watkins, 1989) is a model-free, off-policy RL algorithm that learns action-value functions without requiring a model of the environment. The Q-function Q(s, a) represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter. The update rule is:

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]
```

Where:
- α is the learning rate (how quickly new information overrides old)
- γ is the discount factor (how much future rewards matter)
- r is the immediate reward
- s' is the next state

**Epsilon-greedy exploration** balances exploitation (choosing the best known action) with exploration (trying random actions to discover better strategies). Epsilon (ε) starts high and decays over time.

## 2.2 Datacenter Energy Optimization

Google's DeepMind demonstrated in 2016 that RL could reduce datacenter cooling energy by 40%. Their system used neural networks to predict the Power Usage Effectiveness (PUE) of a datacenter and recommended optimal cooling configurations.

Microsoft's Project Natick explored underwater datacenters for natural cooling. Facebook's Autoscale system uses ML to predict load patterns and pre-provision resources.

## 2.3 Carbon-Aware Computing

Carbon-aware computing is an emerging field that considers the carbon intensity of electricity when making computational decisions. Key concepts include:

- **Carbon intensity mapping**: Different energy sources produce different amounts of CO2 per kWh
- **Temporal shifting**: Delaying non-urgent workloads to periods of low carbon intensity
- **Spatial shifting**: Moving workloads to regions with cleaner energy grids
- **Demand shaping**: Adapting computation intensity based on grid carbon signals

## 2.4 Gap Analysis

Existing systems typically optimize for a single objective (cost OR carbon OR performance). GreenGold addresses the gap by using RL to simultaneously optimize across all three dimensions, learning the trade-offs through experience rather than hand-coded rules.

---

# Chapter 3: System Architecture

## 3.1 High-Level Architecture

The system consists of three major components:

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE DASHBOARD (Browser)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ KPI Cards│  │ Charts   │  │ Server   │  │ RL Agent   │  │
│  │ (6 cards)│  │ (2 live) │  │ Grid (10)│  │ Panel      │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│                    ▲ polls every 2 seconds                   │
└────────────────────┼────────────────────────────────────────┘
                     │ JSON API
┌────────────────────┼────────────────────────────────────────┐
│              Flask Server + RL Agent (app.py)                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ REST API    │  │ Q-Learning   │  │ Physics Engine    │  │
│  │ 5 endpoints │  │ Agent        │  │ Temp simulation   │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────── OFFLINE PIPELINE ────────────────────────┐
│  data_generator → analytics_agent → migration_planner       │
└─────────────────────────────────────────────────────────────┘
```

## 3.2 Component Details

### 3.2.1 Simulation Engine (app.py)

The simulation models a datacenter with the following hierarchy:

- **5 Regions**: Iceland, USA (Virginia), Denmark, India (Mumbai), France
- **10 Servers**: Randomly distributed across regions
- **80 GPU Chips**: 8 per server (NVIDIA H100, NVIDIA A100, AMD MI300X)

Each chip maintains state:
- `base_load`: The assigned workload percentage (modified by agent actions)
- `utilization`: Actual utilization = base_load + small noise
- `temp`: Temperature derived from physics model
- `task`: Current workload type (LLM Training, Image Gen, Vector Search, Idle)
- `anomaly`: Boolean flag (temp > 82°C or util > 92%)
- `cooldown`: Ticks remaining in post-migration cooldown period

### 3.2.2 Physics-Based Temperature Model

Unlike naive random-walk temperature models, GreenGold uses a physics-based approach:

```python
target_temp = ambient + 15 + utilization × 0.65 + noise(±1.5°C)
actual_temp = 0.6 × previous_temp + 0.4 × target_temp
```

Key properties of this model:
- Temperature is **causally linked** to utilization — reducing workload causes cooling
- The exponential moving average (60/40 blend) creates smooth, realistic transitions
- Cooling takes 2-3 ticks to fully manifest (thermal inertia)
- Ambient temperature creates regional differences (Iceland chips run ~23°C cooler than Mumbai at same load)
- Small noise (±1.5°C) simulates measurement variation and micro-fluctuations

### 3.2.3 Workload Dynamics

Each environment tick:
1. Base load drifts ±0.5% (simulating gradual demand changes)
2. Utilization = base_load + noise(±2%) + occasional burst/lull
3. 5% chance of burst (+5-15% spike)
4. 3% chance of lull (-5-10% drop)
5. Temperature recomputed from new utilization
6. VRAM usage follows utilization proportionally
7. Anomaly flags updated

### 3.2.4 Flask Web Server

The server exposes 5 REST API endpoints:

| Endpoint | Method | Response | Side Effect |
|----------|--------|----------|-------------|
| `/` | GET | Dashboard HTML | None |
| `/api/summary` | GET | KPI metrics JSON | Triggers one agent tick |
| `/api/metrics` | GET | Last 60 time-series points | None |
| `/api/rl_log` | GET | Last 50 agent decisions | None |
| `/api/agent_stats` | GET | Episode/reward statistics | None |
| `/api/servers` | GET | All server/chip state | None |

The `/api/summary` endpoint is the clock driver — each call advances the simulation by one tick, runs one agent step, and returns updated metrics.

### 3.2.5 Frontend Dashboard (index.html)

Single-page application built with vanilla HTML/CSS/JavaScript and Chart.js. No build step or framework required.

Components:
- **6 KPI Cards**: Avg utilization, avg temperature, anomaly count, total chips, greenest region, agent episodes
- **Utilization Chart**: Real-time line chart with 30-point sliding window
- **Temperature & Anomaly Chart**: Dual-axis chart (temperature left, anomaly count right)
- **Server Grid**: 10 interactive cards with utilization bars, temperature bars, chip status dots
- **RL Agent Panel**: Episode count, average reward, total reward, exploration rate
- **Decision Log**: Scrollable list of last 15 agent actions with color-coded badges
- **Region Table**: Static reference for carbon and cost profiles

Visual feedback system:
- Green/yellow/orange/red color gradient for chip health
- Blue pulsing dots for chips in post-migration cooldown
- Green flash animation when server temperature drops significantly
- Red flash animation when server temperature rises significantly
- Carbon badges (green/red) on each server card

---

# Chapter 4: Reinforcement Learning Agent Design

## 4.1 Problem Formulation as MDP

The datacenter optimization problem is formulated as a Markov Decision Process (MDP):

- **Agent**: The workload optimizer
- **Environment**: The simulated datacenter (10 servers, 80 chips)
- **State**: Discretized representation of datacenter health
- **Actions**: Workload management operations
- **Reward**: Multi-objective score combining utilization, carbon, temperature, and anomalies
- **Episode**: Each API poll = one episode (continuous, non-episodic task)

## 4.2 State Space Design

The continuous datacenter state is discretized into 48 states using three dimensions:

| Dimension | Raw Value | Discretization | Levels |
|-----------|-----------|---------------|--------|
| Avg Utilization | 0-100% | `min(3, floor(avg_util / 25))` | 4: [0-25, 25-50, 50-75, 75-100] |
| Avg Temperature | 20-115°C | `min(3, floor((avg_temp - 20) / 20))` | 4: [20-40, 40-60, 60-80, 80+] |
| Avg Carbon | 0-650 gCO2 | `min(2, floor(avg_carbon / 200))` | 3: [0-200, 200-400, 400+] |

State index = `u × 12 + t × 3 + k` where u, t, k are the bucket indices.

This compact representation (48 states) allows fast Q-table convergence while capturing the essential dimensions of datacenter health.

## 4.3 Action Space Design

The agent has 4 possible actions:

### Action 0: Hold
- **Precondition**: Any state
- **Effect**: No changes to workload distribution
- **Reward logic**: +0.6 if system is stable (0 anomalies, <5 hot chips); -0.3 if anomalies exist; 0.0 otherwise
- **Rationale**: Rewards the agent for recognizing when the system is healthy and intervention is unnecessary

### Action 1: Shift Hottest Chip
- **Precondition**: Hottest chip must be >65°C and not in cooldown
- **Effect**: 
  - Identifies the hottest chip across all servers
  - Finds the coolest server (by average chip temperature)
  - Transfers 50% of the hot chip's base_load to the least-loaded chip on the cool server
  - Source chip enters 5-tick cooldown
  - Both chips get immediate temperature recalculation
- **Reward**: Proportional to temperature reduction + base bonus of 0.5
- **Rationale**: Directly addresses thermal emergencies by redistributing heat

### Action 2: Rebalance Utilization
- **Precondition**: Must have chips >70% AND chips <30% utilization (not in cooldown)
- **Effect**:
  - Identifies up to 4 overloaded/underloaded pairs
  - Transfers load to equalize toward 45% each
  - Both chips get temperature recalculation
- **Reward**: 0.5 per pair rebalanced + 0.3 base
- **Rationale**: Prevents utilization extremes that cause thermal issues

### Action 3: Migrate High-Carbon
- **Precondition**: Must have servers with carbon >300 AND servers with carbon <50
- **Effect**:
  - Finds busiest chip on highest-carbon server
  - Transfers 60% of its workload to least-loaded chip on lowest-carbon server
  - Source chip enters cooldown, may become Idle
  - Temperature recalculated for both
- **Reward**: Proportional to carbon differential + 0.5 base
- **Rationale**: Reduces overall carbon footprint by shifting compute to green regions

## 4.4 Reward Function

The reward function combines action-specific rewards with a global environment assessment:

```
total_reward = action_reward + environment_reward
```

Environment reward components:

| Component | Formula | Range | Weight |
|-----------|---------|-------|--------|
| Utilization score | `0.5 - abs(avg_util - 55) / 100` | [-0.5, +0.5] | 1.0 |
| Carbon penalty | `-(avg_carbon / 650) × 0.8` | [-0.8, 0] | 0.8 |
| Temperature penalty | `-max(0, (avg_temp - 70) / 40) × 1.5` | [-1.7, 0] | 1.5 |
| Anomaly penalty | `-anomaly_count × 0.15` | [-12, 0] | 0.15/chip |

The reward function was designed with these principles:
- **Temperature has the heaviest weight** — preventing hardware damage is critical
- **Anomalies are punished per-chip** — each anomalous chip degrades the reward
- **55% utilization is the sweet spot** — enough throughput without excessive heat
- **Carbon is a persistent penalty** — the agent always benefits from greener operations

## 4.5 Learning Algorithm

### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]
```

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|--------------|
| Learning rate (α) | 0.15 | Moderate — fast enough to learn from sparse events, stable enough to converge |
| Discount factor (γ) | 0.95 | High — future rewards matter (temperature changes take 2-3 ticks) |
| Initial ε | 0.40 | 40% random exploration — ensures all actions are tried early |
| ε decay | 0.997 per episode | Reaches ~5% by episode 600 — sufficient exploration before exploitation |
| Minimum ε | 0.05 | Always 5% random — prevents the agent from getting stuck in local optima |

### Exploration Strategy

The ε-greedy policy decays over three phases:

| Phase | Episodes | ε Range | Agent Behavior |
|-------|----------|---------|----------------|
| Exploration | 0-200 | 0.40-0.22 | Mostly random; tries all actions; builds initial Q-table |
| Transition | 200-600 | 0.22-0.05 | Increasingly exploits learned policy; anomalies begin dropping |
| Exploitation | 600+ | ~0.05 | Nearly optimal; maintains stable system; acts only when needed |

## 4.6 Convergence and Stability

The agent converges to a stable policy typically within 500-700 episodes. Key convergence indicators:

1. **Anomaly count → 0**: The agent learns to proactively prevent thermal events
2. **Average temperature stabilizes ~50-55°C**: Well below the 82°C anomaly threshold
3. **Action distribution shifts to Hold**: Once the system is optimized, maintenance is the correct policy
4. **Average reward stabilizes ~+0.63**: Consistent positive reward indicates learned optimal behavior

---

# Chapter 5: Offline Analytics Pipeline

## 5.1 Data Generation (data_generator.py)

Generates synthetic but realistic datacenter telemetry:

- **Infrastructure registry**: 10 servers assigned to 5 regions with energy profiles
- **Telemetry data**: 50 timesteps × 10 servers × 8 chips = 4,000 telemetry snapshots
- **Anomaly injection**: 2% anomaly rate with three types:
  - Thermal Anomaly: Cooling failure (+40-60°C spike, utilization throttled to 20%)
  - VRAM Leak: Memory exceeds capacity (101-110%)
  - Zombie Process: Stuck at 99.9% utilization with near-zero VRAM

## 5.2 Analytics Engine (analytics_agent.py)

Performs four analysis tasks:

1. **Data Enrichment**: Merges infrastructure + telemetry, computes carbon footprint per chip (`utilization × 0.4kW × carbon_intensity`), thermal delta, Z-score anomaly detection
2. **Utilization Forecasting**: Linear regression per location predicting future congestion; flags regions >80%
3. **Rule-Based Agent**: Answers natural language queries about deployment, health, and sustainability
4. **Visualization**: 2×2 matplotlib dashboard (carbon by region, chip temp by model, forecasted utilization, anomaly distribution)

## 5.3 Migration Planner (migration_planner.py)

Self-healing migration system:

1. Identifies chips with Z-score > 2.5 (thermal stress)
2. Finds candidate target servers (sufficient VRAM, thermally stable, different server)
3. Ranks candidates by cost and carbon intensity
4. Estimates migration cost: $0.10/GB base + 50% surcharge for cross-region
5. Falls back to PUBLIC_CLOUD_BURST ($5.00 flat) if no internal capacity
6. Generates migration_report.csv with financial audit

---

# Chapter 6: Implementation Details

## 6.1 Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Backend | Python | 3.10+ | Core programming language |
| Web Framework | Flask | 3.0 | REST API server |
| Frontend | HTML/CSS/JS | ES6 | Dashboard UI |
| Charts | Chart.js | 4.4 | Real-time data visualization |
| Data Analysis | Pandas | 2.0+ | Offline data manipulation |
| ML | scikit-learn | 1.3+ | Linear regression forecasting |
| Visualization | Matplotlib/Seaborn | 3.7+/0.12+ | Offline analytics charts |
| Production Server | Gunicorn | 21.0+ | WSGI deployment |

## 6.2 Key Implementation Decisions

### Why Tabular Q-Learning (not Deep RL)?

1. **48-state space is small enough** for a lookup table — neural networks would be overkill
2. **Guaranteed convergence** — tabular Q-learning converges to optimal Q* given sufficient exploration
3. **Interpretability** — the Q-table can be directly inspected to understand agent preferences
4. **Speed** — no GPU training required; runs in microseconds per step

### Why Physics-Based Temperature (not Random Walk)?

The original implementation used `temp += random.uniform(-2, 3)` which:
- Had upward bias (+0.5°C/tick average) causing runaway heating
- Was decoupled from utilization — reducing load had no cooling effect
- Made agent actions meaningless — the environment ignored them

The physics model `temp = f(utilization, ambient)` ensures:
- Causal link between workload and temperature
- Agent actions produce observable, learnable effects
- Regional ambient differences create meaningful optimization opportunities

### Why Single-Page Dashboard (not React/Vue)?

1. **Zero build step** — just open index.html
2. **No dependencies** — only Chart.js from CDN
3. **Simple deployment** — Flask serves the single file
4. **Fast iteration** — edit and refresh

## 6.3 Project File Manifest

| File | Lines | Bytes | Role |
|------|-------|-------|------|
| app.py | ~310 | ~11KB | Flask server + RL agent + simulation |
| index.html | ~750 | ~23KB | Dashboard (CSS + JS inline) |
| data_generator.py | 179 | 7.1KB | Synthetic data generator |
| analytics_agent.py | 249 | 9.1KB | Offline analytics pipeline |
| migration_planner.py | 123 | 4.3KB | Migration plan generator |
| main.py | 63 | 2.1KB | Pipeline orchestrator |
| requirements.txt | 7 | 120B | Python dependencies |
| Procfile | 1 | 45B | Deployment config |

---

# Chapter 7: Results and Evaluation

## 7.1 Agent Learning Performance

After running the system, the following metrics were observed:

| Metric | Initial (Episode 0) | After Learning (Episode 1300+) |
|--------|---------------------|-------------------------------|
| Average Temperature | ~82°C | ~53°C |
| Anomaly Count | 35+ chips | 0 chips |
| Average Reward | Variable (-1 to +2) | Stable at +0.63 |
| Total Reward | 0 | 846+ |
| Exploration Rate (ε) | 0.40 | 0.05 |
| Dominant Action | Random | Hold (system stabilized) |

## 7.2 Temperature Stabilization

The physics-based model ensures agent actions produce measurable effects:

- **Before shift action**: Hottest chip at ~95°C, high anomaly count
- **After shift action**: Source chip cools to ~55-60°C within 2-3 ticks
- **Target server**: Warms proportionally but remains within safe range (cooler ambient helps)

## 7.3 Carbon Optimization

The Migrate High-Carbon action successfully transfers workloads:
- From: USA (Virginia, 650 gCO2) and India (Mumbai, 450 gCO2)
- To: Iceland (20 gCO2), Denmark (11 gCO2), France (12 gCO2)
- Carbon reduction per migration: 60-640 gCO2/kWh differential

## 7.4 Agent Policy Analysis

Once trained, the agent exhibits intelligent behavior:
- **Holds** when system is stable (0 anomalies, moderate temps)
- **Shifts** when any chip exceeds thermal threshold
- **Rebalances** when utilization variance is high
- **Migrates** when high-carbon servers are heavily loaded and green servers have capacity

---

# Chapter 8: Challenges and Limitations

## 8.1 Simulation Fidelity

- The simulation uses simplified thermal dynamics; real datacenter cooling involves complex airflow, liquid cooling, and HVAC interactions
- Network latency for cross-region migration is not modeled
- Workload profiles are synthetic and may not reflect real ML training patterns

## 8.2 Scalability

- The 48-state tabular Q-table works for 10 servers but would not scale to thousands
- For production scale, Deep Q-Networks (DQN) or Policy Gradient methods would be needed
- The single-threaded Flask server processes one tick per API call

## 8.3 Multi-Objective Optimization

- The reward function uses fixed weights for carbon, cost, temperature, and utilization
- Different operators may have different priorities (e.g., prioritize cost over carbon)
- A Pareto-optimal approach could expose the trade-off frontier

## 8.4 Real-World Integration

- Real datacenters have hardware APIs, not simulated state
- Migration involves actual data transfer, downtime, and consistency concerns
- Energy grid carbon intensity changes in real-time (not static per region)

---

# Chapter 9: Future Work

1. **Deep RL**: Replace tabular Q-learning with DQN or PPO for larger state/action spaces
2. **Multi-Agent RL**: One agent per region coordinating via communication channels
3. **Real Carbon API**: Integrate with electricityMap or WattTime for live carbon intensity
4. **Predictive Workloads**: Use LSTM/Transformer to forecast incoming workloads
5. **Cost Optimization**: Add dynamic electricity pricing with time-of-use rates
6. **Hardware Integration**: Connect to NVIDIA DCGM or IPMI for real GPU telemetry
7. **Kubernetes Integration**: Deploy as a K8s operator that manages real pod scheduling

---

# Chapter 10: Conclusion

GreenGold demonstrates that Reinforcement Learning can effectively manage the multi-objective optimization problem inherent in modern GPU datacenter operations. The Q-Learning agent, starting with zero knowledge, learns through trial and error to:

1. **Reduce thermal anomalies from 35+ to 0** by proactively shifting workloads before chips overheat
2. **Stabilize average temperature from 82°C to 53°C** through intelligent workload distribution
3. **Optimize carbon footprint** by migrating compute from fossil-fuel regions to renewable-energy regions
4. **Maintain utilization balance** by spreading load across underutilized servers

The physics-based simulation ensures that agent actions produce realistic, observable effects — when workload is removed from a server, it visibly cools down. This causal link is essential for the RL agent to learn meaningful policies.

The live web dashboard provides real-time visibility into the agent's decision-making process, making the system transparent and interpretable. Operators can observe the agent's learning phases, verify its actions, and understand its reasoning through the decision log.

The project establishes a foundation for deploying RL-based workload optimization in production datacenters, with clear paths for scaling to larger environments through deep RL, real-time carbon data integration, and hardware API connectivity.

---

# References

1. Watkins, C.J.C.H., Dayan, P. (1992). Q-Learning. Machine Learning, 8(3), 279-292.
2. Sutton, R.S., Barto, A.G. (2018). Reinforcement Learning: An Introduction. MIT Press.
3. Evans, R., Gao, J. (2016). DeepMind AI Reduces Google Data Centre Cooling Bill by 40%. DeepMind Blog.
4. Radovanovic, A. et al. (2022). Carbon-Aware Computing for Datacenters. Nature, 598, 392-396.
5. Patterson, D. et al. (2021). Carbon Emissions and Large Neural Network Training. arXiv:2104.10350.
6. Lazic, N. et al. (2018). Data center cooling using model-predictive control. NeurIPS.
7. International Energy Agency (2024). Data Centres and Data Transmission Networks. IEA Report.
8. Masanet, E. et al. (2020). Recalibrating global data center energy-use estimates. Science, 367(6481).

---

# Appendix A: Region Energy Profiles

| Region | Energy Source | Carbon (gCO2/kWh) | Cost ($/kWh) | Ambient Temp (°C) | Grid Capacity (MW) |
|--------|-------------|-------------------|-------------|-------------------|-------------------|
| Iceland | Geothermal/Hydro | 20 | 0.05 | 5 | 500 |
| Denmark | Wind | 11 | 0.28 | 9 | 800 |
| France | Nuclear | 12 | 0.18 | 14 | 3000 |
| USA (Virginia) | Coal/Gas Mix | 650 | 0.12 | 18 | 2000 |
| India (Mumbai) | Solar/Coal Mix | 450 | 0.09 | 28 | 1200 |

# Appendix B: Q-Learning Pseudocode

```
Initialize Q(s, a) = 0 for all states s, actions a
Set ε = 0.40, α = 0.15, γ = 0.95

For each episode:
    1. Run physics simulation tick (update temps, utils)
    2. Observe state s = encode(avg_util, avg_temp, avg_carbon)
    3. With probability ε: choose random action a
       Otherwise: choose a = argmax_a Q(s, a)
    4. Execute action a (shift/rebalance/migrate/hold)
    5. Observe reward r and new state s'
    6. Update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    7. Decay: ε ← max(0.05, ε × 0.997)
```

# Appendix C: API Response Schemas

### GET /api/summary
```json
{
    "avg_util": 51.5,
    "avg_temp": 53.3,
    "anomaly_count": 0,
    "total_chips": 80,
    "greenest_loc": "Denmark",
    "epsilon": 0.05
}
```

### GET /api/servers (per server)
```json
{
    "server_id": "SRV-001",
    "location": "Iceland",
    "ambient": 5,
    "carbon": 20,
    "chips": [
        {
            "chip_id": "SRV-001-C00",
            "utilization": 45.2,
            "base_load": 44.8,
            "temp": 52.1,
            "anomaly": false,
            "task": "LLM Training",
            "cooldown": 0
        }
    ]
}
```

### GET /api/agent_stats
```json
{
    "episode": 1318,
    "avg_reward": 0.642,
    "total_reward": 846.095,
    "actions_taken": 1318
}
```
