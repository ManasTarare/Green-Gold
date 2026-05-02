"""
analytics_agent.py — RL-driven offline analytics for datacenter telemetry.

Replaces the old LinearRegression approach with DQN agent policy queries.
The trained RL agent evaluates each historical state and recommends actions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")

from datacenter_env import DatacenterEnv, ACTION_NAMES, OBS_DIM, N_ACTIONS
from rl_agent import DQNAgent


# ==========================================
# 1. DATA INGESTION & ENRICHMENT
# ==========================================
def load_and_enrich_data():
    """
    Loads infrastructure and telemetry CSVs, merges them,
    and adds carbon, thermal, and anomaly columns.
    """
    try:
        infra_df     = pd.read_csv('infrastructure_registry.csv')
        telemetry_df = pd.read_csv('telemetry_log.csv')
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        print("   Run data_generator.py first to create the CSV files.")
        raise

    telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'])
    df = pd.merge(telemetry_df, infra_df, on='server_id', how='inner')

    carbon_map = {
        'Geothermal/Hydro': 20, 'Nuclear': 12, 'Wind': 11,
        'Solar/Coal Mix': 450, 'Coal/Gas Mix': 650
    }
    df['carbon_intensity'] = df['energy_source'].map(carbon_map)
    df['total_carbon_g'] = (df['utilization_percent'] / 100) * 0.4 * df['carbon_intensity']
    df['thermal_delta']  = df['chip_temp_c'] - df['ambient_temp_c']

    vram_cap_col  = 'vram_capacity_gb' if 'vram_capacity_gb' in df.columns else 'vram_capacity'
    vram_used_col = 'vram_used_gb'     if 'vram_used_gb'     in df.columns else 'vram_used'
    df['vram_free'] = df[vram_cap_col] - df[vram_used_col]

    stats = df.groupby(['chipset_model', 'task'])['chip_temp_c'].agg(['mean', 'std']).reset_index()
    df    = df.merge(stats, on=['chipset_model', 'task'], suffixes=('', '_avg'))
    df['temp_z_score'] = (df['chip_temp_c'] - df['mean']) / df['std'].replace(0, np.nan)
    df['temp_z_score'] = df['temp_z_score'].fillna(0)

    print(f"✅ Data loaded and enriched: {len(df):,} records across {df['location'].nunique()} regions.")
    return df


# ==========================================
# 2. RL-BASED ANALYSIS (replaces LinearRegression)
# ==========================================
def run_rl_analysis(df, agent, env):
    """
    Use the trained DQN agent to analyze historical telemetry.
    For each timestep, builds an observation and queries the agent's policy.

    Returns:
        pd.DataFrame: Per-timestamp recommended actions and Q-values.
    """
    results = []
    timestamps = sorted(df['timestamp'].unique())

    for ts in timestamps:
        snapshot = df[df['timestamp'] == ts]

        # Build a normalized observation matching the 26-dim OBS_DIM layout
        avg_util    = snapshot['utilization_percent'].mean() / 100.0
        avg_temp    = snapshot['chip_temp_c'].mean() / 120.0
        max_temp    = snapshot['chip_temp_c'].max() / 120.0
        std_util    = snapshot['utilization_percent'].std(ddof=0) / 50.0
        std_temp    = snapshot['chip_temp_c'].std(ddof=0) / 40.0
        avg_load    = avg_util  # approximate base load ≈ utilization

        # FIX: use column indexing, not .get() which is a dict method
        if 'anomaly_tag' in snapshot.columns:
            n_anom = (snapshot['anomaly_tag'] != 'None').sum() / len(snapshot)
        else:
            n_anom = 0.0

        n_hot       = (snapshot['chip_temp_c'] > 78).sum() / len(snapshot)
        n_idle      = (snapshot['task'] == 'Idle').sum() / len(snapshot)
        n_cooldown  = 0.0   # not tracked in static telemetry
        n_overload  = (snapshot['utilization_percent'] > 70).sum() / len(snapshot)
        n_underload = (snapshot['utilization_percent'] < 30).sum() / len(snapshot)

        avg_carbon  = snapshot['carbon_intensity'].mean() / 650.0 if 'carbon_intensity' in snapshot.columns else 0.3
        std_carbon  = snapshot['carbon_intensity'].std(ddof=0) / 325.0 if 'carbon_intensity' in snapshot.columns else 0.2
        avg_ambient = snapshot['ambient_temp_c'].mean() / 30.0 if 'ambient_temp_c' in snapshot.columns else 0.5

        # Per-server avg utilizations (up to 10 servers → dims 16-25)
        srv_utils = (
            snapshot.groupby('server_id')['utilization_percent'].mean() / 100.0
        ).values.tolist()

        # FIX: build obs array with exactly OBS_DIM=26 dimensions
        base = [
            avg_util,    # 0
            std_util,    # 1
            avg_temp,    # 2
            max_temp,    # 3
            std_temp,    # 4
            avg_load,    # 5
            n_anom,      # 6
            n_hot,       # 7
            n_idle,      # 8
            n_cooldown,  # 9
            n_overload,  # 10
            n_underload, # 11
            avg_carbon,  # 12
            std_carbon,  # 13
            avg_ambient, # 14
            0.5,         # 15  (tick progress unknown for offline data)
        ]
        # Pad or truncate server utils to exactly 10 slots (dims 16-25)
        srv_part = (srv_utils + [avg_util] * 10)[:10]
        obs = np.array(base + srv_part, dtype=np.float32)
        assert len(obs) == OBS_DIM, f"obs dim mismatch: {len(obs)} != {OBS_DIM}"
        obs = np.clip(obs, 0.0, 2.0)

        q_vals = agent.get_q_values(obs)
        best_action = int(np.argmax(q_vals))

        results.append({
            'timestamp':    ts,
            'recommended':  ACTION_NAMES[best_action],
            'best_q':       round(max(q_vals), 4),
            'avg_util':     round(avg_util * 100, 1),
            'avg_temp':     round(avg_temp * 120, 1),
            'anomaly_rate': round(n_anom * 100, 1),
        })

    print(f"✅ RL analysis complete: {len(results)} timesteps evaluated.")
    return pd.DataFrame(results)


# ==========================================
# 3. DASHBOARD VISUALIZATION
# ==========================================
def show_dashboard(df, rl_results=None, savefig_only=False):
    """
    Displays a 2×2 dashboard with RL-driven insights.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#71717a')
        ax.xaxis.label.set_color('#71717a')
        ax.yaxis.label.set_color('#71717a')
        ax.title.set_color('#e4e4e7')
    fig.suptitle("🖥️  GreenGold DQN Intelligence Dashboard", fontsize=16, fontweight='bold', color='#e4e4e7')

    # Plot 1: Carbon Footprint by Region
    carbon_agg = df.groupby('location', as_index=False)['total_carbon_g'].mean()
    sns.barplot(data=carbon_agg, x='location', y='total_carbon_g', palette='magma', ax=axes[0, 0])
    axes[0, 0].set_title("Carbon Footprint (gCO2) by Region")
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Plot 2: Chip Temp by Model
    sns.boxplot(data=df, x='chipset_model', y='chip_temp_c', palette='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title("Chip Temperature by Model")

    # Plot 3: RL Action Recommendations (if available)
    if rl_results is not None and not rl_results.empty:
        action_counts = rl_results['recommended'].value_counts()
        colors = ['#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#06b6d4', '#71717a']
        axes[1, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
                       colors=colors[:len(action_counts)], textprops={'color': '#e4e4e7'})
        axes[1, 0].set_title("DQN Agent Recommended Actions")
    else:
        axes[1, 0].text(0.5, 0.5, 'No RL analysis data', ha='center', va='center', color='#71717a')
        axes[1, 0].set_title("DQN Agent Actions (N/A)")

    # Plot 4: Anomaly Distribution
    if 'anomaly_tag' in df.columns:
        anomaly_counts = df['anomaly_tag'].value_counts()
        axes[1, 1].pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%',
                       colors=sns.color_palette('Set2', len(anomaly_counts)),
                       textprops={'color': '#e4e4e7'})
    axes[1, 1].set_title("Anomaly Distribution")

    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("✅ Dashboard saved as 'dashboard.png'.")
    if not savefig_only:
        plt.show()
    plt.close(fig)


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print("   DQN DATACENTER ANALYTICS ENGINE")
    print("=" * 55)

    merged_df = load_and_enrich_data()

    # Create agent and try to load trained model
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    env   = DatacenterEnv()

    import os
    if os.path.exists('model.pt'):
        agent.load('model.pt')
        print("✅ Loaded trained DQN model.")
    elif os.path.exists('model.npz'):
        agent.load('model.npz')
        print("✅ Loaded trained DQN model.")
    else:
        print("⚠️  No trained model found. Using untrained agent.")

    rl_results = run_rl_analysis(merged_df, agent, env)
    print("\nRL Recommended Actions per Timestep:")
    print(rl_results.to_string(index=False))

    show_dashboard(merged_df, rl_results)
