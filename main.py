"""
main.py — Run the full DQN datacenter intelligence pipeline.

Order:
  1. Generate infrastructure registry + telemetry CSVs
  2. Create DatacenterEnv + DQNAgent
  3. Train the agent (2000 steps)
  4. Save trained model
  5. Run RL-based offline analysis
  6. Generate RL-driven migration report
  7. Save dashboard
"""

from data_generator import generate_infrastructure, generate_telemetry
from datacenter_env import DatacenterEnv, OBS_DIM, N_ACTIONS
from rl_agent import DQNAgent, train_agent
from analytics_agent import load_and_enrich_data, run_rl_analysis, show_dashboard
from migration_planner import generate_migration_report, print_financial_summary


def main():
    print("=" * 55)
    print("   STEP 1: DATA GENERATION")
    print("=" * 55)
    generate_infrastructure(num_servers=10, seed=42)
    generate_telemetry(num_steps=50, chips_per_server=8, anomaly_rate=0.02, seed=42)

    print()
    print("=" * 55)
    print("   STEP 2: DQN AGENT TRAINING")
    print("=" * 55)
    env   = DatacenterEnv(seed=42)
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        lr=1e-3,
        gamma=0.95,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=2000,
    )

    print("Training DQN agent for 2000 steps...")
    history = train_agent(env, agent, num_steps=2000, log_every=200)

    # Save trained model
    try:
        agent.save("model.pt")
        print("✅ Model saved as 'model.pt'.")
    except Exception:
        agent.save("model.npz")
        print("✅ Model saved as 'model.npz'.")

    print()
    print("=" * 55)
    print("   STEP 3: RL-DRIVEN ANALYTICS")
    print("=" * 55)
    df = load_and_enrich_data()
    rl_results = run_rl_analysis(df, agent, env)

    print("\nRL Recommended Actions (sample):")
    print(rl_results.head(10).to_string(index=False))

    print()
    print("=" * 55)
    print("   STEP 4: RL MIGRATION PLANNER")
    print("=" * 55)
    report = generate_migration_report(df, agent)
    if not report.empty:
        display = report.copy()
        display["Migration_Cost"] = display["Migration_Cost"].apply(lambda x: f"${x:.2f}")
        print(display.to_string(index=False))
        print_financial_summary(report)
        report.to_csv("migration_report.csv", index=False)
        print("✅ migration_report.csv saved.")

    print()
    print("=" * 55)
    print("   STEP 5: DASHBOARD")
    print("=" * 55)
    show_dashboard(df, rl_results, savefig_only=True)

    print()
    print("=" * 55)
    print("   ✅ PIPELINE COMPLETE")
    print(f"   Agent trained for {agent.get_stats()['train_steps']} gradient steps")
    print(f"   Final ε = {agent.epsilon:.4f}")
    print(f"   Total reward = {agent.total_reward:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
