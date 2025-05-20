# ORCA-RL

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

We present ORCA-RL, a reinforcement-learning platform for multi-agent navigation, featuring a high-performance C++ core built on RVO2 and a Gymnasium-compliant Python API compatible with standard RL libraries such as Stable-Baselines3. The platform includes five scenario types with obstacles and collision-avoidance challenges, supports six distinct navigation behavior profiles, and offers an extensible API for creating custom environments.

## Environment Suite

Each environment presents a distinct challenge scenario. Animated videos are included to visualize the setup and how an agent can go through the challenge.

### Circle

Agents are arranged evenly around a circle. Each agent’s goal is the position directly opposite its start on the circumference. They must coordinate to cross the center without collisions.

<video src="./documentation/video/circle.mp4" controls width="480"></video>


**Key Features**  
- **N-agents** on a circle  
- Goals on opposite points  
- Dense crossing at center  


### Incoming

A single “ego” agent must traverse a crowd moving in the opposite direction. All other agents have goals on the side the ego agent starts from, so they move head-on toward the ego.

<div align="center">
  <video src="./documentation/video/incoming.mp4" controls width="480"></video>
</div>

**Key Features**  
- One agent vs. oncoming crowd  
- Opposing-velocity interactions  
- Tests reactive collision avoidance  

### Perpendicular 1

Simulates a one-way street crossing. The ego agent needs to cross a perpendicularly moving crowd to reach its goal on the opposite sidewalk.

[▶️ View Perpendicular 1 Environment Video (mp4)](./documentation/video/perp1.mp4)

**Key Features**  
- Ego crosses a single-direction flow  
- Crowd moves orthogonally  
- Emulates pedestrian crosswalk  

### Perpendicular 2

Like Perpendicular 1, but the crowd moves in both directions (two-way street). The ego agent must find gaps in bidirectional traffic to cross.

[▶️ View Perpendicular 2 Environment Video (mp4)](./documentation/video/perp2.mp4)

**Key Features**  
- Bi-directional pedestrian flow  
- Increased gap-finding complexity  
- Tests planning under mixed traffic  

### Two Paths

The ego agent chooses between two parallel corridors leading to its goal. Each path has different crowd density. Other agents move against the ego’s direction.

[▶️ View Two Paths Environment Video (mp4)](./documentation/video/two_paths.mp4)

**Key Features**  
- Choice between high-density vs. low-density path  
- Counter-flow agents in both corridors  
- Evaluates path-selection strategies  


## Table of Contents

- [ORCA-RL](#orca-rl)
  - [Description](#description)
  - [Environment Suite](#environment-suite)
    - [Circle](#circle)
    - [Incoming](#incoming)
    - [Perpendicular 1](#perpendicular-1)
    - [Perpendicular 2](#perpendicular-2)
    - [Two Paths](#two-paths)
  - [Table of Contents](#table-of-contents)
  - [Installation and Usage](#installation-and-usage)
  - [License](#license)

## Installation and Usage

> **⚠️ Project Status:** This repository is **under active development** as part of my thesis work. Usage instructions and documentation will be updated over time. In the meantime, experienced Python users can still clone the code, install dependencies, and run the simulations with minimal effort.  
>  
> **Prerequisite:** Be sure to install my `PYRVO2-RL` package (authored by me) from GitHub: https://github.com/felipecer/PYRVO2-RL. It’s not included as a submodule here, so you’ll need to install it separately (e.g. `pip install git+https://github.com/felipecer/PYRVO2-RL.git`) before running any environment.  


## License

This project is licensed under the [MIT License](LICENSE).