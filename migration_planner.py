"""
migration_planner.py — RL-driven self-healing migration planner.

Uses the trained DQN agent to recommend migration actions instead of
static Z-score thresholding. The agent evaluates each anomalous state
and recommends the optimal action.
"""

import pandas as pd
import numpy as np

from datacenter_env import OBS_DIM, N_ACTIONS, ACTION_NAMES
from rl_agent import DQNAgent


# ==========================================
# 1. RL-BASED MIGRATION PLANNER
# ==========================================
def generate_migration_report(df, agent):
    """
    Uses the DQN agent to generate a migration plan for anomalous chips.
    Instead of Z-score rules, the agent's policy determines what to do.
    """
    vram_used_col = 'vram_used_gb' if 'vram_used_gb' in df.columns else 'vram_used'

    # Identify anomalous chips using Z-score > 2.5 as DETECTION (not decision)
    anomalies = df[df['temp_z_score'] > 2.5].copy()

    if anomalies.empty:
        print("✅ No anomalies found. System is healthy — no migration needed.")
        return pd.DataFrame()

    plan_data = []
    for _, row in anomalies.iterrows():
        # Build observation vector from this chip's context
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = row['utilization_percent'] / 100.0
        obs[2] = row['chip_temp_c'] / 120.0
        obs[3] = row['chip_temp_c'] / 120.0  # max temp = this chip
        obs[6] = 0.5  # anomaly present
        obs[7] = 1.0 if row['chip_temp_c'] > 78 else 0.0
        obs[12] = (row['carbon_intensity'] if 'carbon_intensity' in row.index else 300) / 650.0
        obs[14] = (row['ambient_temp_c'] if 'ambient_temp_c' in row.index else 15) / 30.0
        obs = np.clip(obs, 0.0, 2.0)

        # Query the DQN agent
        q_vals = agent.get_q_values(obs)
        best_action = int(np.argmax(q_vals))
        action_name = ACTION_NAMES[best_action]
        confidence  = round(max(q_vals), 3)

        # Determine migration target based on agent's action
        vram_used = row[vram_used_col]
        if best_action in [1, 3, 6]:  # Shift, Migrate Carbon, Boost Green
            # Find safe candidates
            candidates = df[
                (df['vram_free'] >= vram_used) &
                (df['temp_z_score'] <= 1.0) &
                (df['server_id'] != row['server_id'])
            ]
            if not candidates.empty:
                target = candidates.sort_values(
                    by=['energy_cost_per_kwh', 'carbon_intensity']).iloc[0]
                target_info = f"{target['server_id']} ({target['location']})"
                base_cost = vram_used * 0.10
                cost = base_cost * 1.5 if target['location'] != row['location'] else base_cost
            else:
                target_info = "PUBLIC_CLOUD_BURST"
                cost = 5.00
        elif best_action == 4:  # Emergency Cool-Down
            target_info = f"{row['server_id']} (IN-PLACE COOLDOWN)"
            cost = 0.50
        else:
            target_info = f"{row['server_id']} (MONITOR)"
            cost = 0.0

        plan_data.append({
            "Anomalous_Chip":  row['chipset_id'],
            "Current_Temp_C":  row['chip_temp_c'],
            "Z_Score":         round(row['temp_z_score'], 2),
            "RL_Action":       action_name,
            "Q_Confidence":    confidence,
            "Shift_To":        target_info,
            "Migration_Cost":  round(cost, 2),
        })

    report = pd.DataFrame(plan_data).drop_duplicates(subset=['Anomalous_Chip'])
    return report


# ==========================================
# 2. FINANCIAL SUMMARY
# ==========================================
def print_financial_summary(report):
    if report.empty:
        return
    total_cost    = report['Migration_Cost'].sum()
    cloud_bursts  = len(report[report['Shift_To'] == 'PUBLIC_CLOUD_BURST'])
    internal_migs = len(report[report['Migration_Cost'] > 0]) - cloud_bursts
    cooldowns     = len(report[report['Shift_To'].str.contains('COOLDOWN', na=False)])

    print("\n" + "=" * 55)
    print("   💰 FINANCIAL AUDIT (RL-Driven)")
    print("=" * 55)
    print(f"  Total Anomalies Addressed : {len(report):,}")
    print(f"  RL-Recommended Migrations : {internal_migs:,}")
    print(f"  In-Place Cooldowns        : {cooldowns:,}")
    print(f"  Cloud Burst (overflow)    : {cloud_bursts:,}")
    print(f"  Total Estimated Cost      : ${round(total_cost, 2):,.2f}")
    print("=" * 55)


# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    from analytics_agent import load_and_enrich_data
    import os

    print("=" * 55)
    print("   RL SELF-HEALING MIGRATION PLANNER")
    print("=" * 55)

    merged_df = load_and_enrich_data()

    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    if os.path.exists('model.pt'):
        agent.load('model.pt')
        print("✅ Loaded trained DQN model.")
    elif os.path.exists('model.npz'):
        agent.load('model.npz')
        print("✅ Loaded trained DQN model.")
    else:
        print("⚠️  No trained model. Using untrained agent.")

    report = generate_migration_report(merged_df, agent)

    if not report.empty:
        print(f"\n🚨 RL MIGRATION PLAN — {len(report)} anomalous chips\n")
        print(report.to_string(index=False))
        print_financial_summary(report)
        report.to_csv("migration_report.csv", index=False)
        print("\n✅ Report saved as 'migration_report.csv'.")
