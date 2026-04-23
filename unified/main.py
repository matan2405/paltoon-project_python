"""
unified/main.py — Unified longitudinal + lateral platoon merge simulation entry point.

Scenarios (mirrors both Longitudinal and lateral modules):
  3 merge positions × 3 driver types = 9 scenarios

  Merge positions:
    back   — ego joins behind the last platoon vehicle  (join_after)
    middle — ego joins between platoon vehicles         (join_middle)
    front  — ego joins ahead of the platoon leader      (join_before)

  Driver types:
    cautious    — lower cruise speed, larger gaps
    normal      — standard IDM parameters
    aggressive  — higher cruise speed, smaller gaps

  join_trigger_time (~25 s) allows all vehicles to accelerate from rest
  to near-cruise speed before the merge decision is evaluated.
"""

import os
import sys

# Force UTF-8 stdout/stderr so that emoji in Longitudinal/config.py prints
# do not crash on Windows cp1255 console encoding.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

from unified.simulation.simulator import UnifiedSimulation
from unified.visualization.plotter import plot_all
from unified.visualization.ieee_figures import (
    plot_authority_sigmoid,
    plot_nash_control_inputs,
    plot_lateral_dsf_heatmap,
    plot_longitudinal_dsf,
    plot_gne_pareto,
)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
SCENARIOS = {
    # ── Join from BACK ───────────────────────────────────────────────────────
    1: {
        'name':                 'Join Back — Cautious Driver',
        'merge_scenario':       'back',
        'num_platoon_vehicles': 3,
        'driver_type':          'cautious',
        'T_sim':                120.0,
        'merge_trigger_time':   30.0,
    },
    2: {
        'name':                 'Join Back — Normal Driver',
        'merge_scenario':       'back',
        'num_platoon_vehicles': 3,
        'driver_type':          'normal',
        'T_sim':                100.0,
        'merge_trigger_time':   25.0,
    },
    3: {
        'name':                 'Join Back — Aggressive Driver',
        'merge_scenario':       'back',
        'num_platoon_vehicles': 3,
        'driver_type':          'aggressive',
        'T_sim':                90.0,
        'merge_trigger_time':   20.0,
    },
    # ── Join in MIDDLE ───────────────────────────────────────────────────────
    4: {
        'name':                 'Join Middle — Cautious Driver',
        'merge_scenario':       'middle',
        'num_platoon_vehicles': 3,
        'driver_type':          'cautious',
        'T_sim':                120.0,
        'merge_trigger_time':   30.0,
    },
    5: {
        'name':                 'Join Middle — Normal Driver',
        'merge_scenario':       'middle',
        'num_platoon_vehicles': 3,
        'driver_type':          'normal',
        'T_sim':                100.0,
        'merge_trigger_time':   25.0,
    },
    6: {
        'name':                 'Join Middle — Aggressive Driver',
        'merge_scenario':       'middle',
        'num_platoon_vehicles': 3,
        'driver_type':          'aggressive',
        'T_sim':                90.0,
        'merge_trigger_time':   20.0,
    },
    # ── Join from FRONT ──────────────────────────────────────────────────────
    7: {
        'name':                 'Join Front — Cautious Driver',
        'merge_scenario':       'front',
        'num_platoon_vehicles': 3,
        'driver_type':          'cautious',
        'T_sim':                120.0,
        'merge_trigger_time':   30.0,
    },
    8: {
        'name':                 'Join Front — Normal Driver',
        'merge_scenario':       'front',
        'num_platoon_vehicles': 3,
        'driver_type':          'normal',
        'T_sim':                100.0,
        'merge_trigger_time':   25.0,
    },
    9: {
        'name':                 'Join Front — Aggressive Driver',
        'merge_scenario':       'front',
        'num_platoon_vehicles': 3,
        'driver_type':          'aggressive',
        'T_sim':                90.0,
        'merge_trigger_time':   20.0,
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(cfg: dict, animate: bool = True):
    print(f"\n{'=' * 65}")
    print(f"  Scenario: {cfg['name']}")
    print(f"{'=' * 65}")

    sim = UnifiedSimulation(
        num_platoon_vehicles=cfg['num_platoon_vehicles'],
        driver_type=cfg['driver_type'],
        merge_scenario=cfg['merge_scenario'],
        T_sim=cfg['T_sim'],
        merge_trigger_time=cfg['merge_trigger_time'],
    )

    data = sim.run()

    suffix = f"_{cfg['merge_scenario']}_{cfg['driver_type']}"

    # IEEE analytical figures
    plot_authority_sigmoid(show=True, save=True)
    plot_nash_control_inputs(data, scenario_name=cfg['name'], show=True, save=True)
    plot_lateral_dsf_heatmap(data, scenario_name=cfg['name'], show=True, save=True)
    plot_longitudinal_dsf(data, scenario_name=cfg['name'], show=True, save=True)
    plot_gne_pareto(scenario_name=cfg['name'], show=True, save=True)

    plot_all(data, title_suffix=suffix, animate=animate)

    return sim, data


def main():
    print("\n" + "=" * 65)
    print("  UNIFIED LONGITUDINAL + LATERAL PLATOON MERGE SIMULATION")
    print("  Nash GNE Shared Control (Li 2019 / Pustilnik & Borrelli 2025)")
    print("=" * 65)
    print("\nSelect a scenario:\n")
    print("  --- Join from BACK ---")
    for k in (1, 2, 3):
        print(f"  {k}. {SCENARIOS[k]['name']}")
    print("  --- Join in MIDDLE ---")
    for k in (4, 5, 6):
        print(f"  {k}. {SCENARIOS[k]['name']}")
    print("  --- Join from FRONT ---")
    for k in (7, 8, 9):
        print(f"  {k}. {SCENARIOS[k]['name']}")
    print("  0. Run all 9 scenarios")

    try:
        choice = int(input("\nEnter choice [0-9]: ").strip())
    except (ValueError, EOFError):
        choice = 2   # default: join back, normal driver

    if choice == 0:
        for k, cfg in SCENARIOS.items():
            run_scenario(cfg, animate=True)
    elif choice in SCENARIOS:
        run_scenario(SCENARIOS[choice], animate=True)
    else:
        print(f"Invalid choice '{choice}', running scenario 2.")
        run_scenario(SCENARIOS[2], animate=True)


if __name__ == '__main__':
    main()
