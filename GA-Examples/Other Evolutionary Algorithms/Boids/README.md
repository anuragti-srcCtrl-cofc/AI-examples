# Boids Simulation

## Overview

**Boids** is a simulation of flocking behavior, inspired by the natural movements of birds, fish, or other animals moving in groups. Unlike typical **genetic algorithms** or other evolutionary techniques, Boids does **not** rely on fitness functions, selection, or mutation. Instead, each individual "boid" (bird-oid object) follows **simple local rules**, and complex, emergent flocking patterns arise naturally from these interactions.

## Key Rules of Boids

Each boid in the simulation follows three primary rules:

1. **Separation** - avoid crowding nearby boids.
2. **Alignment** - match the velocity (direction and speed) of nearby boids.
3. **Cohesion** - move toward the average position of nearby boids.

Despite each boid following these simple rules locally, their combined behavior produces realistic flocking, schooling, or swarming patterns — an example of **emergent behavior** in multi-agent systems.

## Features of This Simulation

* Adjustable **Separation, Alignment, and Cohesion** via sliders.
* Toggle **mouse as predator/prey** to see boids flee or ignore it.
* **Random goal points** generated at intervals to guide the flock.
* Visual indicators: red circle shows predator avoidance radius; red dots show random goals.
* **Reset button** to restart the simulation.

## How It Differs from Genetic Algorithms

* **Boids:** Each agent acts independently with simple rules; emergent patterns appear from local interactions.
* **Genetic Algorithms:** Population evolves over generations based on a fitness function, selection, and mutation. Boids demonstrates natural flocking **without optimization or evolution**.

## Running the Simulation

1. **Clone the repository**:

```bash
git clone <this repo link>
cd boids-simulation
```

2. **Install dependencies** (recommended to use a virtual environment):

```bash
pip install pygame dearpygui
```

3. **Run the simulation**:

```bash
python boids_simulation.py
```

* Use the sliders and buttons in the **DearPyGui window** to adjust behavior.
* Click **Start Random Goals** to enable moving goals.
* Click **Reset Boids** to reset positions and disable goals/mouse avoidance.

---
