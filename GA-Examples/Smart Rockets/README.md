# Smart Rockets - Genetic Algorithm Pathfinding

Rockets evolve to find optimal paths to a target using genetic algorithms and physics simulation.

**Inspired by Jer Thorp's Smart Rockets:** http://www.blprnt.com/smartrockets/

## Code Structure

### DNA Class (`DNA`)
Represents genetic sequence as an array of force vectors.

**Key attributes:**
- `genes` - Array of 2D force vectors (one per frame of lifetime)
- `maxforce` - Maximum magnitude of any force vector (0.1)

**Key methods:**
- `crossover()` - Single-point crossover combining parent force arrays
- `mutate()` - Randomly replaces force vectors with new random forces

Each gene is a steering force applied to the rocket at a specific frame.

### Rocket Class (`Rocket`)
Individual rocket with physics simulation and genetic DNA.

**Physics properties:**
- `position`, `velocity`, `acceleration` - Standard physics vectors
- Updates using Euler integration each frame

**Genetic properties:**
- `dna` - DNA object containing force sequence
- `fitness` - Performance score (inverse square of distance to target)
- `gene_counter` - Tracks which force to apply next

**Key methods:**
- `calculate_fitness()` - Uses formula: `1 / (distance²)` - closer = exponentially higher fitness
- `run()` - Apply next DNA force, update physics, check if target reached
- `check_target()` - Detects if within 12 pixels of target

### Population Class (`Population`)
Manages rocket population and genetic algorithm cycle.

**Key methods:**
- `live()` - All rockets execute one frame of behavior
- `calculate_fitness()` - Evaluate all rockets after lifetime expires
- `selection()` - Build mating pool using fitness-proportionate selection
- `reproduction()` - Create new generation via parent selection, crossover, mutation

**Cycle per generation:**
1. Live for N frames (lifetime)
2. Calculate fitness based on final distance to target
3. Selection (build mating pool)
4. Reproduction (create new generation)
5. Repeat

### GUI (`SmartRocketsGUI`)
Visual simulation with interactive target placement.

**Features:**
- Click canvas to move target (system adapts in real-time)
- Rockets drawn as triangles oriented by velocity
- Green rockets hit target, gray rockets still searching
- Real-time stats: generation number, frames remaining, max fitness

## Parameters

- **Population Size**: Number of rockets per generation (default: 50)
- **Mutation Rate**: Probability each force vector mutates (default: 1%)
- **Lifetime**: Frames each generation lives (default: 300)

## How Evolution Works

**Generation 1:** Rockets apply random forces, move chaotically

**Generation 5:** Some rockets show bias toward target direction

**Generation 20:** Most rockets find direct paths, optimize timing

**Generation 50+:** Population converges on optimal force sequences

The fitness function (inverse square of distance) heavily rewards proximity, causing rapid evolution toward target-seeking behavior.

## Key Concepts

**DNA as Force Sequence**: Unlike phrase matching, DNA here is an action sequence - a pre-planned series of steering forces applied over time.

**Physics Integration**: Genetic algorithm combined with real physics simulation (velocity, acceleration, position updates).

**Real-time Adaptation**: Move target during evolution - population adapts to new target location within a few generations.

---

## Installation & Running

### Prerequisites
- Python 3.7 or higher
- `tkinter` (usually included with Python)

### Run the Program

```bash
# Clone the repository
git clone https://github.com/anuragti-srcCtrl-cofc/AI-examples.git
cd smart-rockets

# Run directly (no additional packages needed)
python smart_rockets.py
```

### Usage

1. On startup, configure population size, mutation rate, and lifetime
2. Click **START** to begin simulation
3. Click anywhere on the canvas to move the target
4. Watch rockets evolve pathfinding strategies over generations
5. Click **PAUSE** to pause/resume
6. Click **NEW CONFIG** to change parameters

### Controls

- **START/PAUSE**: Control simulation
- **NEW CONFIG**: Change parameters and restart
- **Click Canvas**: Move target to mouse position

The simulation displays:
- Red circle: Target location
- Green circle: Start position
- Gray triangles: Rockets searching for target
- Green triangles: Rockets that reached target
- Stats: Generation count, frames remaining, maximum fitness score