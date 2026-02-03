# Decentralized RL-based Transmission Power Control in VANETs

This project implements a decentralized Reinforcement Learning (RL) framework to optimize **Transmission Power Control (TPC)** in Vehicular Ad-hoc Networks (VANETs). Using a combination of **SUMO** for traffic dynamics, **OMNeT++** and **Veins** for network simulation, and **TensorFlow** for the RL agents.

## 🌟 Research Objective
The goal is to maintain a fully connected vehicular network (Global Connectivity) while minimizing energy consumption by dynamically adjusting each vehicle's transmission radius. Each vehicle acts as an independent agent that learns to balance the trade-off between battery life and network connectivity using a custom **Hamiltonian-based reward function**.

## 🛠 Tech Stack
- **Simulation Engine:** [OMNeT++ 6.3.0](https://omnetpp.org/)
- **Network Framework:** [INET 4.5.4](https://inet.omnetpp.org/)
- **VANET Framework:** [Veins 5.3.1](https://veins.car2x.org/)
- **Traffic Simulator:** [SUMO 1.22](https://www.eclipse.org/sumo/)
- **RL Brain:** [Python 3.12.8](https://www.python.org/) with [TensorFlow 2.18.0](https://wwww.tensorflow.org/)
- **Interface:** [TraCI](https://sumo.dlr.de/docs/TraCI.html) (Traffic Control Interface)

## 📂 Project Structure
```text
VaNet/
│
├── AgeNet/                     # The Library
│   ├── __init__.py
│   │
│   ├── environment/            # layer 1 – world & map
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── simple.py
│   │   ├── sumo.py
│   │   └── geometry.py
│   │
│   ├── physics/                # layer 2 – physical state & motion
│   │   ├── __init__.py
│   │   ├── mobility.py
│   │   └── state.py
│   │
│   ├── communication/          # layer 3 – network & interaction
│   │   ├── __init__.py
│   │   ├── channel.py
│   │   ├── radio.py
│   │   ├── neighbor_finder.py
│   │   └── reques_policy.py
│   │
│   │── learning/               # layer 4 – intelligence & control
│   │   ├── __init__.py
│   │   ├── brain.py
│   │   ├── hamiltonian.py
│   │   ├── radius_controller.py
│   │   └── state.py
│   │
│   │── core/                   # agent + time evolution
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── simulator.py
│   │
│   │── analysis/               # post-processing & evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── topology.py
│   │   └── visualization.py
│   │
│   └── experiments/            # experiment orchestration
│       ├── __init__.py
│       └── exporters.py
│
├── Sumo/                       # Traffic Configuration
│   ├── maps/                   # Zanjan City .net.xml files
│   ├── routes/                 # Vehicle route definitions (.rou.xml)
│   ├── polygons/               # Obstacles and buildings (.poly.xml)
│   ├── radiation/              # adiation Pattern (.xml)
│   ├── physics/                # Physical Layer (.xml)
│   └── SumoScenario/           # Main Files (*.xml)
│       ├── simulation.sumocfg
│       └── simulation.launchd.xml
│
├── omnetpp/                    # Network Simulation (C++/NED)
│   ├── src/                    # Custom RLNode & NetworkServer logic
│   ├── ned/                    # Network topology definitions
│   └── omnetpp.ini             # Physical layer & MAC parameters
│
├── Results/                    # Simulation logs & Excel exports
├── Saved Model/                # Pre-trained .keras models
│
└── main.ipynb                  # Main Execution Notebook
```

## ⚙️ Logic & Hamiltonian Reward 

Each agent (vehicle) aims to minimize a local Hamiltonian function:

$$
H = \sum_i H_i = \sum_i \left[ \alpha_{1} k_i^{2} + \alpha_{2} k_i^{3} + \alpha_{3} r_i^{2} + \alpha_{4} \sum_{j(j \ne i)} \frac{A_{ij}}{r_{ij}} \right]
$$

- **State**: Local vehicle density, current transmission power, and neighbor count.
- **Action**: Increase, Decrease, or Maintain transmission radius.
- **Reward**: Based on the reduction of the Hamiltonian value and maintenance of the global connection path.


## 📊 Performance Metrics
The system evaluates the following:
- **Connectivity Ratio**: Percentage of vehicles connected to the main cluster.
- **Power Consumption**: Average transmission power per node.
- **Convergence**: RL training stability over episodes.
- **Path Availability**: Existence of multi-hop paths between any two nodes.

