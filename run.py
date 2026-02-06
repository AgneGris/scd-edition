from pathlib import Path
import scd

# === Settings ===
input_file = "emg.mat"
config_name = "surface"
config_file=Path("configs.json")

# === Paths ===
input_path = Path("data/input") / input_file
output_path = Path("data/output") / f"{input_path.stem}_{config_name}.pkl"
output_path.parent.mkdir(parents=True, exist_ok=True)

# === Run ===
print(f"Running decomposition: {input_file}")
dictionary, timestamps = scd.train(input_path, config_name=config_name, config_file=config_file)

# === Save ===
scd.save_results(output_path, dictionary)
print(f"Done! Found {len(timestamps)} motor units")
print(f"Saved to: {output_path}")