# S-Scenario Quick Reference

## Quick Start Commands

```bash
# 1. Run baseline (fixed-time signals)
python baseline_scenario_s.py

# 2. Train RL agent (300k steps, ~2-4 hours)
python train_scenario_s.py

# 3. Test trained agent
python test_scenario_s.py

# 4. Visualize results
python plot_scenario_s.py
```

## Files Created

| File | Purpose |
|------|---------|
| `ambulance_s.rou.xml` | S-shaped ambulance route (J4→J6) |
| `traffic_dense.rou.xml` | Heavy traffic flows (300+ vehicles) |
| `draft02_scenario_s.sumocfg` | SUMO configuration |
| `train_scenario_s.py` | Training script (300k steps) |
| `test_scenario_s.py` | Testing script with metrics |
| `baseline_scenario_s.py` | Baseline comparison |
| `plot_scenario_s.py` | Visualization script |

## Output Files

- `models_s/` - Trained models and checkpoints
- `traffic_result_s.csv` - Training metrics
- `baseline_result_s.txt` - Baseline results (JSON)
- `optimized_result_s.txt` - RL agent results (JSON)
- `s_scenario_results.png` - Visualization

## Key Features

- **S-Route**: Ambulance enters north of J4, turns right at J4 onto E3, turns right at J6 onto E5
- **Dual Control**: PPO agent controls both J4 and J6 intersections
- **Heavy Traffic**: 300+ civilian vehicles
- **Enhanced Metrics**: Per-intersection timing, route segments, civilian impact
- **6 Visualizations**: Learning curve, comparisons, per-intersection, route timing, traffic volume

## Verification Status

✅ SUMO configuration tested (132 vehicles in 100s test)
✅ All files created successfully
✅ No conflicts with existing files
✅ Ready to run
