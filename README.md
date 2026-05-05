# Otter USV Expert Simulation

Minimal expert-data simulation stack for a Maritime Robotics Otter USV in 2D obstacle-rich scenes.

## Features

- Otter USV 3-DOF surge-sway-yaw dynamics.
- Twin-propeller control inputs.
- Random static and dynamic obstacle scenarios.
- Sampling-based NMPC-style expert controller.
- Pygame 2D visualization.
- Expert trajectory export for later planning or learning work.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run Demo

```powershell
python -m usv_sim.run_demo --view --seed 0
```

## Generate Dataset

```powershell
python -m usv_sim.generate_dataset --num 10 --out data/otter_samples.npz --seed 0
```

The saved dataset includes:

- `states`: `(N, 121, 7)` as `[x, y, sin(psi), cos(psi), u, v, r]`
- `controls`: `(N, 120, 2)` as left/right propeller shaft speeds
- `env`: `(N, 57)` static/dynamic obstacle and goal encoding

## Visual Dashboard

```powershell
python -m usv_sim.dashboard
```

Open `http://127.0.0.1:8765` to browse all expert trajectories, filter by obstacle difficulty,
replay one selected USV trajectory, and generate new trajectories by difficulty and count.
New dashboard-generated samples are saved to `data/dashboard_generated.npz`.
