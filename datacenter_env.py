"""
datacenter_env.py — Gym-style Datacenter RL Environment.

Provides a proper environment abstraction with:
  - reset()  → observation vector
  - step(action) → (observation, reward, done, info)
  - Continuous observation space (26-dim normalized vector)
  - 7 discrete actions
  - Physics-based thermal model
"""

import random
import numpy as np

# ── Region Profiles ──────────────────────────────────────────────
LOCATIONS = [
    {"name": "Iceland",        "source": "Geothermal/Hydro", "cost": 0.05, "ambient": 5,  "carbon": 20},
    {"name": "USA (Virginia)", "source": "Coal/Gas Mix",     "cost": 0.12, "ambient": 18, "carbon": 650},
    {"name": "Denmark",        "source": "Wind",             "cost": 0.28, "ambient": 9,  "carbon": 11},
    {"name": "India (Mumbai)", "source": "Solar/Coal Mix",   "cost": 0.09, "ambient": 28, "carbon": 450},
    {"name": "France",         "source": "Nuclear",          "cost": 0.18, "ambient": 14, "carbon": 12},
]

NUM_SERVERS      = 10
CHIPS_PER_SERVER = 8
TOTAL_CHIPS      = NUM_SERVERS * CHIPS_PER_SERVER
N_ACTIONS        = 7
ACTION_NAMES     = [
    "Hold",
    "Shift Hottest Chip",
    "Rebalance Utilization",
    "Migrate High-Carbon",
    "Emergency Cool-Down",
    "Consolidate Idle",
    "Boost Green Utilization",
]

# Observation dimension (see get_observation for layout)
OBS_DIM = 26


class DatacenterEnv:
    """Gym-style environment for datacenter workload optimization."""

    EPISODE_LENGTH = 500  # ticks before done=True; enables varied episode starts

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.servers = []
        self.tick_count = 0
        self.episode_reward = 0.0
        self.reset()

    # ── Reset ────────────────────────────────────────────────────
    def reset(self):
        """Initialize all servers/chips and return the first observation."""
        # NOTE: intentionally NOT re-seeding here — each episode should differ
        self.servers = []
        for i in range(NUM_SERVERS):
            loc = self.rng.choice(LOCATIONS)
            chips = []
            for c in range(CHIPS_PER_SERVER):
                base_load = self.rng.uniform(20, 75)
                task = self.rng.choice(["LLM Training", "Image Gen", "Vector Search", "Idle"])
                if task == "Idle":
                    base_load = self.rng.uniform(3, 10)
                util = base_load + self.rng.uniform(-2, 2)
                coeff = 0.3 if task == "Idle" else 0.65
                temp = loc["ambient"] + 15 + util * coeff + self.rng.uniform(-2, 2)
                chips.append({
                    "chip_id":     f"SRV-{i+1:03d}-C{c:02d}",
                    "model":       self.rng.choice(["NVIDIA-H100", "NVIDIA-A100", "AMD-MI300X"]),
                    "base_load":   round(base_load, 1),
                    "utilization": round(max(0, min(100, util)), 1),
                    "temp":        round(temp, 1),
                    "vram_used":   round(self.rng.uniform(8, 70), 1),
                    "vram_cap":    80,
                    "task":        task,
                    "anomaly":     False,
                    "cooldown":    0,
                })
            self.servers.append({
                "server_id": f"SRV-{i+1:03d}",
                "location":  loc["name"],
                "source":    loc["source"],
                "cost":      loc["cost"],
                "ambient":   loc["ambient"],
                "carbon":    loc["carbon"],
                "chips":     chips,
            })
        self.tick_count = 0
        self.episode_reward = 0.0
        return self.get_observation()

    # ── Observation ──────────────────────────────────────────────
    def get_observation(self):
        """Return a 26-dim normalized continuous state vector."""
        all_utils = [c["utilization"] for s in self.servers for c in s["chips"]]
        all_temps = [c["temp"]        for s in self.servers for c in s["chips"]]
        all_loads = [c["base_load"]   for s in self.servers for c in s["chips"]]
        carbons   = [s["carbon"]      for s in self.servers]
        ambients  = [s["ambient"]     for s in self.servers]

        n_anomalies  = sum(1 for s in self.servers for c in s["chips"] if c["anomaly"])
        n_hot        = sum(1 for s in self.servers for c in s["chips"] if c["temp"] > 78)
        n_idle       = sum(1 for s in self.servers for c in s["chips"] if c["task"] == "Idle")
        n_cooldown   = sum(1 for s in self.servers for c in s["chips"] if c["cooldown"] > 0)
        n_overloaded = sum(1 for s in self.servers for c in s["chips"] if c["base_load"] > 70)
        n_underload  = sum(1 for s in self.servers for c in s["chips"] if c["base_load"] < 30)

        # Per-server avg utils (10 values, normalized) — fills dims 16-25
        srv_utils = [np.mean([c["utilization"] for c in s["chips"]]) / 100.0 for s in self.servers]

        obs = np.array([
            # Global aggregates (normalized to ~[0,1])
            np.mean(all_utils) / 100.0,            # 0: mean utilization
            np.std(all_utils) / 50.0,               # 1: util spread
            np.mean(all_temps) / 120.0,              # 2: mean temperature
            max(all_temps) / 120.0,                  # 3: max temperature
            np.std(all_temps) / 40.0,                # 4: temp spread
            np.mean(all_loads) / 100.0,              # 5: mean base load
            n_anomalies / TOTAL_CHIPS,               # 6: anomaly ratio
            n_hot / TOTAL_CHIPS,                     # 7: hot chip ratio
            n_idle / TOTAL_CHIPS,                    # 8: idle chip ratio
            n_cooldown / TOTAL_CHIPS,                # 9: cooldown ratio
            n_overloaded / TOTAL_CHIPS,              # 10: overloaded ratio
            n_underload / TOTAL_CHIPS,               # 11: underloaded ratio
            np.mean(carbons) / 650.0,                # 12: avg carbon (normalized)
            np.std(carbons) / 325.0,                 # 13: carbon spread
            np.mean(ambients) / 30.0,                # 14: avg ambient
            self.tick_count / self.EPISODE_LENGTH,   # 15: progress (time signal)
        ] + srv_utils                                # 16-25: per-server util (10 values)
        , dtype=np.float32)

        return np.clip(obs, 0.0, 2.0)

    # ── Physics Tick ─────────────────────────────────────────────
    def _physics_tick(self):
        """Advance one tick of datacenter physics."""
        for srv in self.servers:
            for c in srv["chips"]:
                # Small organic workload fluctuation
                burst = 0
                if self.rng.random() < 0.05:
                    burst = self.rng.uniform(5, 15)
                elif self.rng.random() < 0.03:
                    burst = self.rng.uniform(-10, -5)

                c["base_load"] = round(max(3, min(92, c["base_load"] + self.rng.uniform(-0.5, 0.5))), 1)
                c["utilization"] = round(max(0, min(100, c["base_load"] + burst + self.rng.uniform(-2, 2))), 1)

                # Physics-based temperature
                target = srv["ambient"] + 15 + c["utilization"] * 0.65 + self.rng.uniform(-1.5, 1.5)
                c["temp"] = round(c["temp"] * 0.6 + target * 0.4, 1)

                # VRAM
                if c["task"] == "Idle":
                    c["vram_used"] = round(self.rng.uniform(0.5, 5), 1)
                else:
                    c["vram_used"] = round(max(0, min(c["vram_cap"],
                        c["vram_cap"] * (c["utilization"] / 100) * self.rng.uniform(0.6, 0.9))), 1)

                c["anomaly"] = c["temp"] > 82 or c["utilization"] > 92

                if c["cooldown"] > 0:
                    c["cooldown"] -= 1

    # ── Actions ──────────────────────────────────────────────────
    def _avg(self, lst):
        return sum(lst) / max(len(lst), 1)

    def _compute_temp(self, chip, ambient):
        target = ambient + 15 + chip["utilization"] * 0.65 + self.rng.uniform(-1.5, 1.5)
        return round(chip["temp"] * 0.6 + target * 0.4, 1)

    def _do_hold(self):
        anoms = sum(1 for s in self.servers for c in s["chips"] if c["anomaly"])
        hot   = sum(1 for s in self.servers for c in s["chips"] if c["temp"] > 78)
        if anoms == 0 and hot < 5:
            return "System stable — no intervention needed.", 0.6
        elif anoms > 0:
            return "System has anomalies — Hold may not be optimal.", -0.3
        else:
            return "Some chips running hot — monitoring.", 0.0

    def _do_shift_hottest(self):
        hot_chip, hot_temp, hot_srv = None, 0, None
        for srv in self.servers:
            for c in srv["chips"]:
                if c["temp"] > hot_temp and c["cooldown"] == 0:
                    hot_temp, hot_chip, hot_srv = c["temp"], c, srv
        if not hot_chip or hot_temp < 65:
            return "No critical thermal events.", 0.1
        cool_srv = min(self.servers, key=lambda x: self._avg([c["temp"] for c in x["chips"]]))
        if cool_srv["server_id"] == hot_srv["server_id"]:
            return "No better target available.", 0.0
        target_chip = min(cool_srv["chips"], key=lambda c: c["base_load"])
        transfer = hot_chip["base_load"] * 0.5
        old_load, old_temp = hot_chip["base_load"], hot_chip["temp"]
        hot_chip["base_load"]   = round(max(5, hot_chip["base_load"] - transfer), 1)
        hot_chip["utilization"] = round(hot_chip["base_load"] + self.rng.uniform(-2, 2), 1)
        hot_chip["cooldown"]    = 5
        if hot_chip["utilization"] < 15:
            hot_chip["task"] = "Idle"
        target_chip["base_load"]   = round(min(90, target_chip["base_load"] + transfer * 0.8), 1)
        target_chip["utilization"] = round(target_chip["base_load"] + self.rng.uniform(-2, 2), 1)
        if target_chip["task"] == "Idle":
            target_chip["task"] = self.rng.choice(["LLM Training", "Image Gen", "Vector Search"])
        hot_chip["temp"]    = self._compute_temp(hot_chip, hot_srv["ambient"])
        target_chip["temp"] = self._compute_temp(target_chip, cool_srv["ambient"])
        new_temp = hot_chip["temp"]
        reward = max(0.5, (old_temp - new_temp) * 0.08) + 0.5
        return (f"Shifted {hot_chip['chip_id']}: {old_temp}°C→{new_temp}°C, "
                f"load {old_load}→{hot_chip['base_load']}%"), reward

    def _do_rebalance(self):
        all_chips = [(s, c) for s in self.servers for c in s["chips"] if c["cooldown"] == 0]
        over  = [(s, c) for s, c in all_chips if c["base_load"] > 70]
        under = [(s, c) for s, c in all_chips if c["base_load"] < 30]
        if not over or not under:
            return "Utilization already balanced.", 0.15
        moved, total_shifted = 0, 0
        for i in range(min(4, len(over), len(under))):
            shift = min(over[i][1]["base_load"] - 45, 45 - under[i][1]["base_load"], 25)
            if shift <= 0:
                continue
            over[i][1]["base_load"]    = round(over[i][1]["base_load"] - shift, 1)
            under[i][1]["base_load"]   = round(under[i][1]["base_load"] + shift, 1)
            over[i][1]["utilization"]  = round(max(0, min(100, over[i][1]["base_load"] + self.rng.uniform(-2, 2))), 1)
            under[i][1]["utilization"] = round(max(0, min(100, under[i][1]["base_load"] + self.rng.uniform(-2, 2))), 1)
            over[i][1]["temp"]  = self._compute_temp(over[i][1], over[i][0]["ambient"])
            under[i][1]["temp"] = self._compute_temp(under[i][1], under[i][0]["ambient"])
            moved += 1
            total_shifted += shift
        if moved == 0:
            return "No significant imbalance to fix.", 0.1
        return f"Rebalanced {moved} pair(s). Shifted {total_shifted:.0f}% load.", moved * 0.5 + 0.3

    def _do_migrate_carbon(self):
        high = [s for s in self.servers if s["carbon"] > 300]
        low  = [s for s in self.servers if s["carbon"] < 50]
        if not high or not low:
            return "No carbon migration opportunity.", 0.05
        src = max(high, key=lambda s: self._avg([c["base_load"] for c in s["chips"]]))
        src_chip = max([c for c in src["chips"] if c["cooldown"] == 0 and c["base_load"] > 30],
                       key=lambda c: c["base_load"], default=None)
        if not src_chip:
            return "No eligible chips for carbon migration.", 0.05
        tgt = min(low, key=lambda s: self._avg([c["base_load"] for c in s["chips"]]))
        tgt_chip = min(tgt["chips"], key=lambda c: c["base_load"])
        transfer = src_chip["base_load"] * 0.6
        old_load = src_chip["base_load"]
        src_chip["base_load"]   = round(max(5, src_chip["base_load"] - transfer), 1)
        src_chip["utilization"] = round(max(0, src_chip["base_load"] + self.rng.uniform(-2, 2)), 1)
        src_chip["cooldown"]    = 5
        if src_chip["base_load"] < 15:
            src_chip["task"] = "Idle"
        tgt_chip["base_load"]   = round(min(90, tgt_chip["base_load"] + transfer * 0.85), 1)
        tgt_chip["utilization"] = round(min(100, tgt_chip["base_load"] + self.rng.uniform(-2, 2)), 1)
        if tgt_chip["task"] == "Idle":
            tgt_chip["task"] = self.rng.choice(["LLM Training", "Image Gen", "Vector Search"])
        src_chip["temp"] = self._compute_temp(src_chip, src["ambient"])
        tgt_chip["temp"] = self._compute_temp(tgt_chip, tgt["ambient"])
        carbon_saved = (src["carbon"] - tgt["carbon"]) * (transfer / 100)
        return (f"Migrated {src_chip['chip_id']}: load {old_load}→{src_chip['base_load']}% "
                f"({src['location']}→{tgt['location']})"), carbon_saved / 50 + 0.5

    def _do_emergency_cooldown(self):
        """Aggressively reduce load on ALL chips above 78°C."""
        cooled = 0
        for srv in self.servers:
            for c in srv["chips"]:
                if c["temp"] > 78 and c["cooldown"] == 0:
                    c["base_load"]   = round(max(5, c["base_load"] * 0.4), 1)
                    c["utilization"] = round(c["base_load"] + self.rng.uniform(-1, 1), 1)
                    c["cooldown"]    = 8
                    c["task"]        = "Idle"
                    c["temp"]        = self._compute_temp(c, srv["ambient"])
                    cooled += 1
        if cooled == 0:
            return "No chips require emergency cooling.", 0.05
        return f"Emergency: Cooled {cooled} chip(s) aggressively.", cooled * 0.7 + 0.3

    def _do_consolidate_idle(self):
        """Move scattered idle workloads to consolidate on fewer servers."""
        idle_chips = [(s, c) for s in self.servers for c in s["chips"]
                      if c["task"] == "Idle" and c["base_load"] > 5 and c["cooldown"] == 0]
        busy_chips = [(s, c) for s in self.servers for c in s["chips"]
                      if c["base_load"] < 60 and c["task"] != "Idle" and c["cooldown"] == 0]
        if not idle_chips or not busy_chips:
            return "No consolidation opportunity.", 0.05
        moved = 0
        for i in range(min(3, len(idle_chips), len(busy_chips))):
            src_s, src_c = idle_chips[i]
            tgt_s, tgt_c = busy_chips[i]
            transfer = src_c["base_load"]
            tgt_c["base_load"]   = round(min(85, tgt_c["base_load"] + transfer * 0.9), 1)
            tgt_c["utilization"] = round(tgt_c["base_load"] + self.rng.uniform(-2, 2), 1)
            src_c["base_load"]   = round(self.rng.uniform(1, 3), 1)
            src_c["utilization"] = round(src_c["base_load"] + self.rng.uniform(-0.5, 0.5), 1)
            tgt_c["temp"] = self._compute_temp(tgt_c, tgt_s["ambient"])
            src_c["temp"] = self._compute_temp(src_c, src_s["ambient"])
            moved += 1
        return f"Consolidated {moved} idle chip(s).", moved * 0.3 + 0.2

    def _do_boost_green(self):
        """Shift work TO green servers to maximize renewable usage."""
        green = [s for s in self.servers if s["carbon"] < 50]
        dirty = [s for s in self.servers if s["carbon"] > 300]
        if not green or not dirty:
            return "No green boost opportunity.", 0.05
        green_under = [(s, c) for s in green for c in s["chips"]
                       if c["base_load"] < 50 and c["cooldown"] == 0]
        dirty_busy  = [(s, c) for s in dirty for c in s["chips"]
                       if c["base_load"] > 40 and c["cooldown"] == 0]
        if not green_under or not dirty_busy:
            return "Green servers at capacity or dirty servers idle.", 0.05
        moved = 0
        for i in range(min(3, len(green_under), len(dirty_busy))):
            tgt_s, tgt_c = green_under[i]
            src_s, src_c = dirty_busy[i]
            transfer = src_c["base_load"] * 0.4
            src_c["base_load"]   = round(max(5, src_c["base_load"] - transfer), 1)
            src_c["utilization"] = round(src_c["base_load"] + self.rng.uniform(-2, 2), 1)
            tgt_c["base_load"]   = round(min(85, tgt_c["base_load"] + transfer * 0.9), 1)
            tgt_c["utilization"] = round(tgt_c["base_load"] + self.rng.uniform(-2, 2), 1)
            src_c["temp"] = self._compute_temp(src_c, src_s["ambient"])
            tgt_c["temp"] = self._compute_temp(tgt_c, tgt_s["ambient"])
            if tgt_c["task"] == "Idle":
                tgt_c["task"] = self.rng.choice(["LLM Training", "Image Gen", "Vector Search"])
            moved += 1
        return f"Boosted green usage: {moved} workload(s) shifted.", moved * 0.5 + 0.4

    # ── Step ─────────────────────────────────────────────────────
    def step(self, action):
        """Execute action, advance physics, return (obs, reward, done, info)."""
        action_fns = [
            self._do_hold,
            self._do_shift_hottest,
            self._do_rebalance,
            self._do_migrate_carbon,
            self._do_emergency_cooldown,
            self._do_consolidate_idle,
            self._do_boost_green,
        ]
        msg, action_reward = action_fns[action]()

        # Advance physics
        self._physics_tick()
        self.tick_count += 1

        # Compute environment reward
        env_reward = self._compute_env_reward()

        # Small action cost for non-Hold (prefer minimal intervention)
        action_cost = 0.0 if action == 0 else -0.05

        total_reward = round(action_reward + env_reward + action_cost, 4)
        self.episode_reward += total_reward

        obs = self.get_observation()
        done = self.tick_count >= self.EPISODE_LENGTH
        info = {
            "action_name": ACTION_NAMES[action],
            "message":     msg,
            "action_reward": action_reward,
            "env_reward":    env_reward,
        }
        return obs, total_reward, done, info

    def _compute_env_reward(self):
        """Multi-objective environment reward."""
        utils   = [c["utilization"] for s in self.servers for c in s["chips"]]
        temps   = [c["temp"]        for s in self.servers for c in s["chips"]]
        carbons = [s["carbon"]      for s in self.servers]
        anoms   = sum(1 for s in self.servers for c in s["chips"] if c["anomaly"])

        avg_util = self._avg(utils)
        avg_temp = self._avg(temps)

        util_score      = 0.5 - abs(avg_util - 55) / 100
        carbon_penalty  = -(self._avg(carbons) / 650) * 0.8
        temp_penalty    = -max(0, (avg_temp - 70) / 40) * 1.5
        anomaly_penalty = -anoms * 0.15

        return round(util_score + carbon_penalty + temp_penalty + anomaly_penalty, 4)

    # ── Utility for external access ──────────────────────────────
    def get_server_data(self):
        """Return server state for API/dashboard."""
        return self.servers

    def get_summary(self):
        """Return KPI summary dict."""
        utils = [c["utilization"] for s in self.servers for c in s["chips"]]
        temps = [c["temp"]        for s in self.servers for c in s["chips"]]
        anoms = sum(1 for s in self.servers for c in s["chips"] if c["anomaly"])
        greenest = min(self.servers, key=lambda x: x["carbon"])
        return {
            "avg_util":      round(self._avg(utils), 1),
            "avg_temp":      round(self._avg(temps), 1),
            "anomaly_count": anoms,
            "total_chips":   TOTAL_CHIPS,
            "greenest_loc":  greenest["location"],
        }
