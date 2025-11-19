# Genetic Algorithm - Evolving Shakespeare

A visual demonstration of genetic algorithms that evolves random strings into the phrase "To be or not to be." using principles of natural selection.

## What is a Genetic Algorithm?

Genetic algorithms solve problems by mimicking biological evolution. Instead of calculating the solution directly, they:

1. Generate random candidate solutions
2. Test how good each solution is (fitness)
3. Select the best candidates to "reproduce"
4. Create new solutions by combining and mutating parent solutions
5. Repeat until an optimal solution emerges

## How This Implementation Works

### The DNA Class
Each `DNA` object represents a potential solution - a string of random characters.

- **Genes**: Array of characters (the actual string)
- **Fitness**: Percentage of characters matching the target (0.0 to 1.0)
- **Crossover**: Combines two parent strings at a random midpoint to create offspring
- **Mutation**: Randomly changes characters to introduce genetic diversity

### The Population Class
Manages the entire population and orchestrates evolution.

**Natural Selection** (Survival of the Fittest):
- Organisms with higher fitness appear more frequently in the "mating pool"
- Example: 100% fit organism gets 100 copies, 50% fit gets 50 copies
- This makes better solutions more likely to reproduce

**Reproduction Cycle** (Each Generation):
1. **Selection**: Build mating pool based on fitness
2. **Crossover**: Pick two random parents, combine their genes
3. **Mutation**: Randomly alter some genes in the child
4. **Replace**: New generation completely replaces the old

### Key Parameters

- **Population Size**: 150 organisms per generation
- **Mutation Rate**: 1% chance per character (0.01)
- **Target Phrase**: "To be or not to be."

Lower mutation rates are more stable but slower. Higher rates introduce more diversity but can be chaotic.

## The Evolution Process

```
Generation 0:  "xK#m!@pL$nRt%qWe^"  (random noise)
Generation 50: "Tozbe xr not mo ye."  (getting closer)
Generation 150: "To be or not to be."  (success!)
```

Average convergence: **100-500 generations**

The algorithm finds the solution through:
- **Selection pressure**: Good solutions reproduce more
- **Crossover**: Combines good traits from multiple solutions  
- **Mutation**: Explores new possibilities randomly

## Why It Works

Genetic algorithms excel at:
- Searching huge solution spaces
- Avoiding local optima (mutation helps escape)
- Finding "good enough" solutions quickly
- Problems where fitness is easy to calculate but solution is hard to compute

They're used in real applications like:
- Circuit design optimization
- Neural network architecture search
- Scheduling problems
- Game AI behavior evolution

---

## Installation & Running

### Prerequisites
- Python 3.7 or higher
- `tkinter` (usually included with Python)

### Run the Program

```bash
# Clone the repository
git clone https://github.com/anuragti-srcCtrl-cofc/AI-examples.git
cd genetic-algorithm-shakespeare

# Run directly (no additional packages needed)
python genetic_algorithm.py
```

### Controls

- **START**: Begin evolution
- **PAUSE**: Pause evolution
- **RESET**: Start over with a new random population

The GUI displays:
- Current best match (green text)
- Fitness progress bar (red → yellow → green)
- Generation count and statistics
- Top 6 candidate solutions

---

## Educational Notes

This is a simplified genetic algorithm for teaching purposes. Production implementations often include:

- **Elitism**: Always keep the best solution(s) across generations
- **Tournament selection**: Alternative to fitness-proportionate selection
- **Adaptive mutation rates**: Decrease mutation as solution improves
- **Multi-point crossover**: Multiple crossover points instead of one

Try experimenting with different:
- Population sizes (50 vs 500)
- Mutation rates (0.001 vs 0.05)
- Target phrases (longer/shorter strings)

Watch how these affect convergence speed and stability!