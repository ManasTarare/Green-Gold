import pandas as pd
import random
from datetime import datetime, timedelta

REGISTRY_FILE = "infrastructure_registry.csv"
TELEMETRY_FILE = "telemetry_log.csv"


# ==========================================
# 1. INFRASTRUCTURE REGISTRY
# ==========================================
def generate_infrastructure(num_servers=10, seed=None):
    """
    Generates a simulated server infrastructure registry and saves it to CSV.

    Args:
        num_servers (int): Number of servers to generate. Default is 10.
        seed (int): Optional random seed for reproducibility.

    Returns:
        pd.DataFrame: The generated infrastructure data.
    """
    if seed is not None:
        random.seed(seed)

    locations = [
        {"country": "Iceland",        "source": "Geothermal/Hydro", "cost": 0.05, "temp": 5,  "cap_mw": 500},
        {"country": "USA (Virginia)", "source": "Coal/Gas Mix",     "cost": 0.12, "temp": 18, "cap_mw": 2000},
        {"country": "Denmark",        "source": "Wind",             "cost": 0.28, "temp": 9,  "cap_mw": 800},
        {"country": "India (Mumbai)", "source": "Solar/Coal Mix",   "cost": 0.09, "temp": 28, "cap_mw": 1200},
        {"country": "France",         "source": "Nuclear",          "cost": 0.18, "temp": 14, "cap_mw": 3000}
    ]

    registry_data = []
    for i in range(num_servers):
        loc = random.choice(locations)
        registry_data.append({
            "server_id":           f"SRV-{i + 1:03d}",
            "location":            loc["country"],
            "energy_source":       loc["source"],
            "energy_cost_per_kwh": loc["cost"],
            "ambient_temp_c":      loc["temp"],
            "grid_capacity_mw":    loc["cap_mw"]
        })

    df = pd.DataFrame(registry_data)

    try:
        df.to_csv(REGISTRY_FILE, index=False)
        print(f"✅ Saved '{REGISTRY_FILE}' with {len(df)} servers.")
    except Exception as e:
        print(f"❌ Failed to save registry: {e}")

    return df


# ==========================================
# 2. TELEMETRY GENERATOR
# ==========================================
def generate_telemetry(num_steps=50, chips_per_server=8, anomaly_rate=0.02, seed=None):
    """
    Generates simulated GPU telemetry data and saves it to CSV.
    Requires infrastructure_registry.csv to exist (run generate_infrastructure first).

    Args:
        num_steps (int): Number of time steps to simulate.
        chips_per_server (int): Number of GPU chips per server.
        anomaly_rate (float): Probability of anomaly per record (0.0 - 1.0).
        seed (int): Optional random seed for reproducibility.

    Returns:
        pd.DataFrame: The generated telemetry data, or None if registry missing.
    """
    if seed is not None:
        random.seed(seed)   # Note: affects global state; prior calls may consume seed values

    try:
        registry = pd.read_csv(REGISTRY_FILE).to_dict('records')
    except FileNotFoundError:
        print("❌ Error: 'infrastructure_registry.csv' not found. Run generate_infrastructure() first!")
        return None

    tasks       = ["LLM Training", "Image Gen", "Vector Search", "Scientific Sim"]
    chip_models = ["NVIDIA-H100", "NVIDIA-A100", "AMD-MI300X"]

    telemetry_data = []
    # Fixed start time for reproducible timestamps across runs
    start_time = datetime(2026, 1, 1, 0, 0, 0)

    for step in range(num_steps):
        current_time = start_time + timedelta(minutes=step * 10)

        for server in registry:
            for c_idx in range(chips_per_server):
                chipset_id = f"{server['server_id']}-CHIP-{c_idx:02d}"
                model      = random.choice(chip_models)
                vram_cap   = 80 if "NVIDIA" in model else 192

                # --- Baseline Logic ---
                is_idle    = random.random() < 0.2
                is_anomaly = random.random() < anomaly_rate

                if is_idle:
                    task        = "Idle"
                    utilization = 0.0
                    vram_used   = 0.0
                    temp        = round(server['ambient_temp_c'] + random.uniform(2, 3), 1)
                else:
                    task        = random.choice(tasks)
                    utilization = random.uniform(30, 98)
                    vram_used   = round(vram_cap * (utilization / 100) * random.uniform(0.7, 0.95), 2)
                    temp        = server['ambient_temp_c'] + 15 + (utilization * 0.5) + random.uniform(-2, 2)

                # --- Anomaly Injection (only on active chips) ---
                anomaly_type = "None"
                if is_anomaly and not is_idle:
                    anomaly_selector = random.random()

                    if anomaly_selector < 0.33:
                        # Thermal Spike (cooling failure + throttling)
                        temp         += random.uniform(40, 60)
                        utilization  *= 0.2
                        anomaly_type  = "Thermal_Anomaly"

                    elif anomaly_selector < 0.66:
                        # Memory Leak (VRAM exceeds capacity)
                        vram_used    = round(vram_cap * random.uniform(1.01, 1.10), 2)
                        anomaly_type = "VRAM_Leak"

                    else:
                        # Zombie Process (high util, no memory activity)
                        utilization  = 99.9
                        vram_used    = 0.1
                        anomaly_type = "Zombie_Process"

                telemetry_data.append({
                    "timestamp":           current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "server_id":           server['server_id'],
                    "chipset_id":          chipset_id,
                    "chipset_model":       model,
                    "task":                task,
                    "vram_capacity_gb":    vram_cap,
                    "vram_used_gb":        round(vram_used, 2),
                    "utilization_percent": round(utilization, 2),
                    "chip_temp_c":         round(temp, 1),
                    "anomaly_tag":         anomaly_type
                })

    df = pd.DataFrame(telemetry_data)

    try:
        df.to_csv(TELEMETRY_FILE, index=False)
        anomaly_count = len(df[df['anomaly_tag'] != 'None'])
        print(f"✅ Saved '{TELEMETRY_FILE}' with {len(df):,} snapshots.")
        print(f"⚠️  Injected {anomaly_count:,} anomalies ({anomaly_count / len(df) * 100:.2f}% of total).")
    except Exception as e:
        print(f"❌ Failed to save telemetry: {e}")

    return df


# ==========================================
# 3. RUN BOTH
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("  STEP 1: Generating Infrastructure Registry")
    print("=" * 50)
    infra_df = generate_infrastructure(num_servers=10, seed=42)

    print()
    print("=" * 50)
    print("  STEP 2: Generating Telemetry Data")
    print("=" * 50)
    telemetry_df = generate_telemetry(num_steps=50, chips_per_server=8, anomaly_rate=0.02, seed=42)

    print()
    print("✅ Both files generated. Ready for analytics pipeline.")
