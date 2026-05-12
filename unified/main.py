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
from unified.simulation.experiments import (
    MonteCarloExperiment,
    QWeightSweepExperiment,
    RWeightSweepExperiment,
    LambdaSweepExperiment,
)
from unified.visualization.experiment_plots import plot_all_experiments


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


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment_mc(n_trials: int = 20, driver_type: str = 'normal'):
    """Proposal A: Monte Carlo — deterministic vs B-IDM vs MA-IDM."""
    print(f"\n{'=' * 65}")
    print(f"  Experiment A: Monte Carlo (N={n_trials}, driver={driver_type})")
    print(f"{'=' * 65}")
    mc = MonteCarloExperiment(n_trials=n_trials, driver_type=driver_type)
    mc.run(verbose=True)
    plot_all_experiments(mc=mc, show=True, save=True)
    return mc


def run_experiment_q_sweep(driver_type: str = 'normal',
                           noise_mode: str = 'gp', n_avg: int = 5):
    """Proposal B1: Q-weight sweep (both longitudinal and lateral)."""
    print(f"\n{'=' * 65}")
    print(f"  Experiment B1: Q-Weight Sweep (driver={driver_type}, "
          f"noise={noise_mode}, n_avg={n_avg})")
    print(f"{'=' * 65}")
    exp = QWeightSweepExperiment(driver_type=driver_type,
                                 noise_mode=noise_mode,
                                 n_avg=n_avg, seed_offset=100)
    exp.run(verbose=True)
    plot_all_experiments(q_sweep=exp, show=True, save=True)
    return exp


def run_experiment_r_sweep(driver_type: str = 'normal',
                           noise_mode: str = 'gp', n_avg: int = 5):
    """Proposal B2: R-weight (effort) sweep (both longitudinal and lateral)."""
    print(f"\n{'=' * 65}")
    print(f"  Experiment B2: R-Weight Sweep (driver={driver_type}, "
          f"noise={noise_mode}, n_avg={n_avg})")
    print(f"{'=' * 65}")
    exp = RWeightSweepExperiment(driver_type=driver_type,
                                 noise_mode=noise_mode,
                                 n_avg=n_avg, seed_offset=200)
    exp.run(verbose=True)
    plot_all_experiments(r_sweep=exp, show=True, save=True)
    return exp


def run_experiment_lambda_sweep(driver_type: str = 'normal',
                                noise_mode: str = 'gp', n_avg: int = 5):
    """Proposal B3: Fixed-λ authority sweep (both longitudinal and lateral)."""
    print(f"\n{'=' * 65}")
    print(f"  Experiment B3: Fixed-λ Sweep (driver={driver_type}, "
          f"noise={noise_mode}, n_avg={n_avg})")
    print(f"{'=' * 65}")
    exp = LambdaSweepExperiment(driver_type=driver_type,
                                noise_mode=noise_mode,
                                n_avg=n_avg, seed_offset=300)
    exp.run(verbose=True)
    exp.print_summary()
    plot_all_experiments(lam_sweep=exp, show=True, save=True)
    return exp


def main():
    print("\n" + "=" * 65)
    print("  UNIFIED LONGITUDINAL + LATERAL PLATOON MERGE SIMULATION")
    print("  Nash GNE Shared Control (Li 2019 / Pustilnik & Borrelli 2025)")
    print("=" * 65)
    print("\nSelect a scenario or experiment:\n")
    print("  --- Base Scenarios ---")
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
    print("\n  --- Experiments ---")
    print(" 10. [A]  Monte Carlo: deterministic vs B-IDM vs MA-IDM (N=20)")
    print(" 11. [B1] Q-weight sweep: tracking precision trade-off")
    print(" 12. [B2] R-weight sweep: control effort decomposition")
    print(" 13. [B3] Fixed-λ sweep: authority allocation effect")
    print(" 14.      Run all 4 experiments (slow — ~hours)")

    try:
        choice = int(input("\nEnter choice [0-14]: ").strip())
    except (ValueError, EOFError):
        choice = 2   # default: join back, normal driver

    if choice == 0:
        for k, cfg in SCENARIOS.items():
            run_scenario(cfg, animate=True)
    elif choice in SCENARIOS:
        run_scenario(SCENARIOS[choice], animate=True)
    elif choice == 10:
        run_experiment_mc()
    elif choice == 11:
        run_experiment_q_sweep()
    elif choice == 12:
        run_experiment_r_sweep()
    elif choice == 13:
        run_experiment_lambda_sweep()
    elif choice == 14:
        run_experiment_mc()
        run_experiment_q_sweep()
        run_experiment_r_sweep()
        run_experiment_lambda_sweep()
    else:
        print(f"Invalid choice '{choice}', running scenario 2.")
        run_scenario(SCENARIOS[2], animate=True)


if __name__ == '__main__':
    main()
