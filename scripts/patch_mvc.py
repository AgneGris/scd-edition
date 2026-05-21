"""
Patch MVC values into existing decomposition pickle files.

Reads MVC values from a channel_config JSON (by aux channel unit name),
then writes them into every .pkl file in the target directory.

Usage:
    uv run python scripts/patch_mvc.py \
        --config  "Z:/path/to/channel_config.json" \
        --output-dir "Z:/path/to/pkl/folder" \
        [--dry-run]
"""

import argparse
import json
import pickle
from pathlib import Path


def load_mvc_map(config_path: Path) -> dict[str, float]:
    """Return {unit_name: mvc_value} from the channel config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)
    mvc_map = {}
    for ch in cfg.get("aux_channels", []):
        unit = ch.get("unit", "").strip()
        mvc = ch.get("mvc")
        if unit and mvc is not None:
            mvc_map[unit] = float(mvc)
    return mvc_map


def patch_file(pkl_path: Path, mvc_map: dict[str, float], dry_run: bool) -> dict:
    """Load one pickle, fill in mvc, save it back. Returns a summary dict."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    aux_channels = data.get("aux_channels", [])
    filled, skipped_already, skipped_no_match = [], [], []

    for ch in aux_channels:
        # Unit name may be nested under 'meta' (raw decomp) or at top level (edited save)
        unit = ch.get("unit") or ch.get("meta", {}).get("unit", "")
        unit = unit.strip()

        if ch.get("mvc") is not None:
            skipped_already.append(unit)
            continue

        mvc_val = mvc_map.get(unit)
        if mvc_val is None:
            skipped_no_match.append(unit)
            continue

        # Write at the top level so it is accessible before and after meta-flattening
        ch["mvc"] = mvc_val
        filled.append(f"{unit} = {mvc_val}")

    summary = {
        "file": pkl_path.name,
        "filled": filled,
        "already_set": skipped_already,
        "no_match": skipped_no_match,
    }

    if filled and not dry_run:
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to channel_config JSON")
    parser.add_argument(
        "--output-dir", required=True, help="Directory with .pkl files to patch"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would change without writing"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    mvc_map = load_mvc_map(config_path)
    print(f"MVC map loaded ({len(mvc_map)} channels):")
    for unit, val in mvc_map.items():
        print(f"  {unit:<20} {val}")
    print()

    pkl_files = sorted(output_dir.glob("*.pkl"))
    if not pkl_files:
        print("No .pkl files found.")
        return

    if args.dry_run:
        print("DRY RUN — no files will be modified.\n")

    total_filled = 0
    for pkl_path in pkl_files:
        summary = patch_file(pkl_path, mvc_map, dry_run=args.dry_run)
        action = "Would patch" if args.dry_run else "Patched"
        if summary["filled"]:
            total_filled += len(summary["filled"])
            print(f"  {action}: {summary['file']}")
            for entry in summary["filled"]:
                print(f"    + {entry}")
        else:
            status = "already set" if summary["already_set"] else "nothing matched"
            print(f"  Skipped ({status}): {summary['file']}")
        if summary["no_match"]:
            print(f"    ! No MVC for: {', '.join(summary['no_match'])}")

    verb = "Would fill" if args.dry_run else "Filled"
    print(f"\n{verb} {total_filled} MVC value(s) across {len(pkl_files)} file(s).")


if __name__ == "__main__":
    main()
