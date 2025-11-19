import tkinter as tk
from tkinter import messagebox
import random
import math

"""
SMART ROCKETS - GENETIC ALGORITHM PATHFINDING
==============================================

This program demonstrates genetic algorithms applied to pathfinding.
Rockets evolve to find the optimal path to a target using force vectors.

KEY CONCEPTS:
-------------
- DNA: Array of force vectors (direction + magnitude) applied each frame
- FITNESS: Based on distance to target and finish time (closer = higher fitness)
- EVOLUTION: Rockets that get closer to target are more likely to reproduce

HOW IT WORKS:
-------------
1. Each rocket has DNA: an array of force vectors (one per frame of life)
2. Every frame, the rocket applies the next force vector in its DNA
3. After lifetime expires, fitness is calculated based on distance and time
4. Best rockets reproduce to create next generation
5. Process repeats, rockets evolve better pathfinding strategies

This is inspired by Jer Thorp's Smart Rockets demonstration.
"""


class Obstacle:
    """
    Represents an obstacle or target as a rectangular region.
    
    Used for both collision detection (obstacles) and goal checking (target).
    """
    
    def __init__(self, x, y, w, h):
        """
        Constructor: Create rectangular obstacle/target.
        
        Args:
            x, y: Top-left corner position
            w, h: Width and height
        """
        self.position = [x, y]
        self.w = w
        self.h = h
    
    def contains(self, point):
        """
        Check if a point is inside this rectangle.
        
        Args:
            point: [x, y] position to check
            
        Returns:
            True if point is inside, False otherwise
        """
        x, y = point
        return (x > self.position[0] and 
                x < self.position[0] + self.w and
                y > self.position[1] and 
                y < self.position[1] + self.h)


class DNA:
    """
    Represents genetic sequence as an array of force vectors.
    
    Each gene is a 2D vector (force) that will be applied to the rocket
    at each frame of its lifetime. This allows the rocket to "steer" itself.
    """
    
    def __init__(self, lifetime, genes=None):
        """
        Constructor: Creates DNA of random force vectors.
        
        Args:
            lifetime: Number of force vectors (genes) in the sequence
            genes: Optional pre-existing gene array (for reproduction)
        """
        self.maxforce = 0.1  # Maximum magnitude of force vectors
        
        if genes:
            # Use provided genes (from crossover)
            self.genes = genes[:]
        else:
            # Generate random force vectors
            # Each vector points in a random direction with random magnitude
            self.genes = []
            for _ in range(lifetime):
                angle = random.uniform(0, 2 * math.pi)
                force = [math.cos(angle), math.sin(angle)]
                magnitude = random.uniform(0, self.maxforce)
                force[0] *= magnitude
                force[1] *= magnitude
                self.genes.append(force)
    
    def crossover(self, partner):
        """
        CROSSOVER - Creates offspring DNA from two parents.
        
        Single-point crossover: pick random midpoint, combine genes from both parents.
        
        Args:
            partner: The other parent DNA
            
        Returns:
            New DNA object with genes from both parents
        """
        child_genes = []
        crossover_point = random.randint(0, len(self.genes) - 1)
        
        for i in range(len(self.genes)):
            if i > crossover_point:
                child_genes.append(self.genes[i][:])
            else:
                child_genes.append(partner.genes[i][:])
        
        return DNA(len(self.genes), child_genes)
    
    def mutate(self, mutation_rate):
        """
        MUTATION - Randomly replaces force vectors.
        
        For each gene, roll the dice. If below mutation rate, replace
        with a completely new random force vector.
        
        Args:
            mutation_rate: Probability (0.0-1.0) that each gene mutates
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                # Generate new random force vector
                angle = random.uniform(0, 2 * math.pi)
                force = [math.cos(angle), math.sin(angle)]
                magnitude = random.uniform(0, self.maxforce)
                force[0] *= magnitude
                force[1] *= magnitude
                self.genes[i] = force


class Rocket:
    """
    Represents a rocket with physics simulation and genetic DNA.
    
    The rocket applies forces from its DNA sequence to navigate toward target.
    Fitness is based on how close it gets to the target and how fast it gets there.
    """
    
    def __init__(self, position, dna, obstacles, target):
        """
        Constructor: Initialize rocket with position and DNA.
        
        Args:
            position: Starting position [x, y]
            dna: DNA object containing force vector sequence
            obstacles: List of obstacle regions for collision detection
            target: Obstacle object representing the target
        """
        # Physics properties
        self.position = position[:]
        self.velocity = [0, 0]
        self.acceleration = [0, 0]
        
        # Appearance
        self.r = 8  # Rocket size
        
        # Genetic properties
        self.dna = dna
        self.fitness = 0
        self.gene_counter = 0  # Which gene we're currently applying
        
        # State tracking
        self.hit_target = False
        self.hit_obstacle = False
        self.obstacles = obstacles
        self.target = target
        
        # Tracking for fitness calculation
        self.record_dist = 10000  # Closest distance to target ever achieved
        self.finish_time = 0  # How many frames until target hit (or lifetime)
    
    def calculate_fitness(self):
        """
        FITNESS FUNCTION - Based on distance to target and finish time.
        
        Formula: fitness = (1 / finish_time * record_dist)^4
        
        Heavily penalizes hitting obstacles (90% fitness loss).
        Heavily rewards reaching target (2x fitness multiplier).
        
        Combines distance and time into a single fitness score.
        Rockets that reach the target quickly and close get highest scores.
        """
        if self.record_dist < 1:
            self.record_dist = 1
        
        if self.finish_time == 0:
            self.finish_time = 1
        
        # Base fitness: inverse of (finish_time * distance)
        self.fitness = 1.0 / (self.finish_time * self.record_dist)
        
        # Make it exponential to heavily favor good solutions
        self.fitness = pow(self.fitness, 4)
        
        # Penalties and bonuses
        if self.hit_obstacle:
            self.fitness *= 0.1  # Lose 90% of fitness hitting obstacle
        
        if self.hit_target:
            self.fitness *= 2  # Twice the fitness for finishing!
            # Also scale up by a multiplier to avoid extremely small numbers
            self.fitness *= 1000
    
    def check_target(self):
        """
        Check if rocket has reached the target and update distance tracking.
        
        Updates record_dist (closest approach to target).
        Sets hit_target flag when target is reached.
        Increments finish_time each frame.
        """
        dx = self.position[0] - self.target.position[0]
        dy = self.position[1] - self.target.position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Track closest approach
        if distance < self.record_dist:
            self.record_dist = distance
        
        # Check if target reached
        if self.target.contains(self.position) and not self.hit_target:
            self.hit_target = True
        else:
            # Count frames until target is hit
            if not self.hit_target:
                self.finish_time += 1
    
    def check_obstacles(self):
        """
        Check if rocket has collided with any obstacle.
        
        Sets hit_obstacle flag on collision. Once set, rocket stops updating.
        """
        for obs in self.obstacles:
            if obs.contains(self.position):
                self.hit_obstacle = True
    
    def apply_force(self, force):
        """
        Apply a force vector to the rocket's acceleration.
        
        Args:
            force: [x, y] force vector to apply
        """
        self.acceleration[0] += force[0]
        self.acceleration[1] += force[1]
    
    def update(self):
        """
        Update physics: apply acceleration to velocity, velocity to position.
        
        Standard Euler integration for simple physics simulation.
        Reset acceleration after applying (forces are instantaneous).
        """
        self.velocity[0] += self.acceleration[0]
        self.velocity[1] += self.acceleration[1]
        
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        
        # Reset acceleration
        self.acceleration[0] = 0
        self.acceleration[1] = 0
    
    def run(self):
        """
        Execute one frame of rocket behavior.
        
        PROCESS:
        1. Check if we've hit the target or obstacle
        2. If not stuck, apply force and update physics
        3. Move to next gene in sequence
        
        Once a rocket hits target or obstacle, it stops updating.
        """
        self.check_target()
        
        if not self.hit_obstacle and not self.hit_target:
            # Apply the current gene (force vector) from DNA
            self.apply_force(self.dna.genes[self.gene_counter])
            
            # Move to next gene, wrap around if at end
            self.gene_counter = (self.gene_counter + 1) % len(self.dna.genes)
            
            # Update physics
            self.update()
            
            # Check for obstacle collision
            self.check_obstacles()
    
    def get_fitness(self):
        return self.fitness
    
    def get_dna(self):
        return self.dna


class Population:
    """
    Manages population of rockets and orchestrates genetic algorithm.
    
    Implements the full GA cycle:
    1. Live: All rockets execute their behavior
    2. Fitness: Calculate how well each rocket performed
    3. Selection: Create mating pool based on fitness
    4. Reproduction: Create new generation through crossover and mutation
    """
    
    def __init__(self, mutation_rate, pop_size, lifetime, start_pos, obstacles, target):
        """
        Initialize rocket population.
        
        Args:
            mutation_rate: Probability of gene mutation (0.0-1.0)
            pop_size: Number of rockets in population
            lifetime: Number of frames each generation lives
            start_pos: Starting position [x, y] for all rockets
            obstacles: List of obstacle regions
            target: Target obstacle object
        """
        self.mutation_rate = mutation_rate
        self.lifetime = lifetime
        self.start_pos = start_pos[:]
        self.obstacles = obstacles
        self.target = target
        self.generations = 0
        
        # Create initial population with random DNA
        self.population = []
        for _ in range(pop_size):
            dna = DNA(lifetime)
            rocket = Rocket(self.start_pos[:], dna, self.obstacles, self.target)
            self.population.append(rocket)
        
        # Mating pool for selection
        self.mating_pool = []
    
    def live(self):
        """
        Run one frame of life for all rockets.
        
        Each rocket executes its behavior (apply force, update physics).
        Calculate fitness in real-time so we can display best rocket live.
        """
        for rocket in self.population:
            rocket.run()
            # Calculate fitness continuously so best rocket is highlighted in real-time
            rocket.calculate_fitness()
    
    def target_reached(self):
        """
        Check if any rocket in the population has reached the target.
        
        Returns:
            True if at least one rocket hit the target
        """
        for rocket in self.population:
            if rocket.hit_target:
                return True
        return False
    
    def calculate_fitness(self):
        """
        Calculate fitness for all rockets in the population.
        
        Called after lifetime expires to evaluate how well each rocket did.
        """
        for rocket in self.population:
            rocket.calculate_fitness()
    
    def get_best_finish_time(self):
        """
        Get the best (shortest) finish time in the population.
        
        Returns the fastest time any rocket reached the target.
        If no rocket reached target, returns lifetime (maximum time).
        """
        best_time = self.lifetime
        for rocket in self.population:
            if rocket.hit_target and rocket.finish_time < best_time:
                best_time = rocket.finish_time
        return best_time
    
    def selection(self):
        """
        NATURAL SELECTION - Build mating pool based on fitness.
        
        Fitness-proportionate selection: rockets with higher fitness
        appear more frequently in the mating pool.
        
        PROCESS:
        1. Find maximum fitness in population
        2. Normalize each rocket's fitness (0.0-1.0)
        3. Add each rocket to pool N times (N proportional to fitness)
        """
        self.mating_pool.clear()
        
        # Find max fitness for normalization
        max_fitness = self.get_max_fitness()
        
        if max_fitness == 0:
            max_fitness = 1
        
        # Build mating pool
        for rocket in self.population:
            # Normalize fitness to 0-1
            normalized = rocket.get_fitness() / max_fitness
            
            # Add to mating pool N times (N based on fitness)
            # Arbitrary multiplier of 100
            n = int(normalized * 100)
            for _ in range(n):
                self.mating_pool.append(rocket)
    
    def reproduction(self):
        """
        REPRODUCTION - Create new generation from mating pool.
        
        PROCESS for each new rocket:
        1. Select two random parents from mating pool
        2. Crossover: combine parent DNA
        3. Mutation: randomly alter some genes
        4. Create new rocket with child DNA
        
        The new generation replaces the old one completely.
        """
        new_population = []
        
        for _ in range(len(self.population)):
            # Select two parents randomly from mating pool
            # (fitter rockets appear more, so more likely to be selected)
            mom = random.choice(self.mating_pool)
            dad = random.choice(self.mating_pool)
            
            # Get their DNA
            mom_dna = mom.get_dna()
            dad_dna = dad.get_dna()
            
            # Crossover: create child DNA
            child_dna = mom_dna.crossover(dad_dna)
            
            # Mutation: random changes
            child_dna.mutate(self.mutation_rate)
            
            # Create new rocket with child DNA
            rocket = Rocket(self.start_pos[:], child_dna, self.obstacles, self.target)
            new_population.append(rocket)
        
        # Replace old population with new generation
        self.population = new_population
        self.generations += 1
    
    def get_generations(self):
        return self.generations
    
    def get_max_fitness(self):
        """Get the highest fitness in the current population."""
        if not self.population:
            return 0
        return max(r.get_fitness() for r in self.population)


class ConfigDialog:
    """Dialog for configuring simulation parameters"""
    
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configuration")
        self.dialog.geometry("500x450")
        self.dialog.configure(bg="#1a1a2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Title
        title = tk.Label(
            self.dialog,
            text="Smart Rockets Configuration",
            font=("Arial", 18, "bold"),
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title.pack(pady=20)
        
        # Parameters frame
        params_frame = tk.Frame(self.dialog, bg="#16213e", relief=tk.RAISED, bd=2)
        params_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Population size
        tk.Label(
            params_frame,
            text="Population Size:",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        self.pop_entry = tk.Entry(
            params_frame,
            font=("Courier", 12),
            bg="#0f0f1e",
            fg="#ffffff",
            insertbackground="#00d4ff"
        )
        self.pop_entry.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.pop_entry.insert(0, "50")
        
        # Mutation rate
        tk.Label(
            params_frame,
            text="Mutation Rate (%):",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(5, 5))
        
        tk.Label(
            params_frame,
            text="Enter 1 for 1%, 5 for 5%, etc.",
            font=("Arial", 9, "italic"),
            bg="#16213e",
            fg="#888"
        ).pack(anchor=tk.W, padx=10)
        
        self.mutation_entry = tk.Entry(
            params_frame,
            font=("Courier", 12),
            bg="#0f0f1e",
            fg="#ffffff",
            insertbackground="#00d4ff"
        )
        self.mutation_entry.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.mutation_entry.insert(0, "1")
        
        # Lifetime
        tk.Label(
            params_frame,
            text="Lifetime (frames):",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(5, 5))
        
        tk.Label(
            params_frame,
            text="How many frames each generation lives",
            font=("Arial", 9, "italic"),
            bg="#16213e",
            fg="#888"
        ).pack(anchor=tk.W, padx=10)
        
        self.lifetime_entry = tk.Entry(
            params_frame,
            font=("Courier", 12),
            bg="#0f0f1e",
            fg="#ffffff",
            insertbackground="#00d4ff"
        )
        self.lifetime_entry.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.lifetime_entry.insert(0, "300")
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg="#1a1a2e")
        button_frame.pack(pady=20)
        
        tk.Button(
            button_frame,
            text="Start Simulation",
            command=self.ok,
            font=("Arial", 12, "bold"),
            bg="#00ff88",
            fg="#1a1a2e",
            activebackground="#00cc6a",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel,
            font=("Arial", 12, "bold"),
            bg="#ff6b6b",
            fg="#ffffff",
            activebackground="#cc5555",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def ok(self):
        try:
            pop_size = int(self.pop_entry.get().strip())
            if pop_size < 10:
                messagebox.showerror("Error", "Population size must be at least 10!")
                return
        except ValueError:
            messagebox.showerror("Error", "Population size must be a valid number!")
            return
        
        try:
            mutation_pct = float(self.mutation_entry.get().strip())
            if mutation_pct < 0 or mutation_pct > 100:
                messagebox.showerror("Error", "Mutation rate must be between 0 and 100!")
                return
            mutation_rate = mutation_pct / 100.0
        except ValueError:
            messagebox.showerror("Error", "Mutation rate must be a valid number!")
            return
        
        try:
            lifetime = int(self.lifetime_entry.get().strip())
            if lifetime < 50:
                messagebox.showerror("Error", "Lifetime must be at least 50 frames!")
                return
        except ValueError:
            messagebox.showerror("Error", "Lifetime must be a valid number!")
            return
        
        self.result = (pop_size, mutation_rate, lifetime)
        self.dialog.destroy()
    
    def cancel(self):
        self.dialog.destroy()


class SmartRocketsGUI:
    """GUI for Smart Rockets genetic algorithm simulation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Rockets - Genetic Algorithm")
        self.root.geometry("800x750")
        self.root.configure(bg="#1a1a2e")
        
        # Simulation parameters
        self.pop_size = None
        self.mutation_rate = None
        self.lifetime = None
        self.population = None
        self.life_counter = 0
        self.running = False
        self.speed = 5  # ms per frame
        
        # Track best fitness and fastest time across all generations
        self.best_fitness_ever = 0.0
        self.record_time = 0  # Fastest time to reach target
        
        # World parameters
        self.canvas_width = 760
        self.canvas_height = 500
        self.target_pos = [self.canvas_width // 2, 30]
        self.start_pos = [self.canvas_width // 2, self.canvas_height - 20]
        
        # Obstacles (including target)
        self.target = Obstacle(self.target_pos[0] - 12, self.target_pos[1] - 12, 24, 24)
        self.obstacles = [Obstacle(self.canvas_width // 2 - 100, self.canvas_height // 2, 200, 10)]
        
        self.setup_ui()
        self.show_config()
    
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title = tk.Label(
            self.root,
            text="SMART ROCKETS",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title.pack(pady=10)
        
        # Canvas for simulation
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#0f0f1e",
            highlightthickness=2,
            highlightbackground="#00d4ff"
        )
        self.canvas.pack(padx=20, pady=10)
        
        # Bind click to move target
        self.canvas.bind("<Button-1>", self.move_target)
        
        # Stats frame
        stats_frame = tk.Frame(self.root, bg="#16213e", relief=tk.RAISED, bd=2)
        stats_frame.pack(padx=20, pady=10, fill=tk.X)
        
        self.gen_label = tk.Label(
            stats_frame,
            text="Generation: 0",
            font=("Arial", 11, "bold"),
            bg="#16213e",
            fg="#ffffff"
        )
        self.gen_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.life_label = tk.Label(
            stats_frame,
            text="Frames Left: 0",
            font=("Arial", 11, "bold"),
            bg="#16213e",
            fg="#ffffff"
        )
        self.life_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.fitness_label = tk.Label(
            stats_frame,
            text="Max Fitness: 0.00",
            font=("Arial", 11, "bold"),
            bg="#16213e",
            fg="#ffffff"
        )
        self.fitness_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.record_label = tk.Label(
            stats_frame,
            text="Record Time: --",
            font=("Arial", 11, "bold"),
            bg="#16213e",
            fg="#ffffff"
        )
        self.record_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="#1a1a2e")
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="START",
            command=self.toggle_simulation,
            font=("Arial", 12, "bold"),
            bg="#00ff88",
            fg="#1a1a2e",
            activebackground="#00cc6a",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2"
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame,
            text="NEW CONFIG",
            command=self.show_config,
            font=("Arial", 12, "bold"),
            bg="#ff6b6b",
            fg="#ffffff",
            activebackground="#cc5555",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Click on canvas to move target • Rockets evolve to find the path",
            font=("Arial", 9, "italic"),
            bg="#1a1a2e",
            fg="#888"
        )
        instructions.pack(pady=5)
    
    def show_config(self):
        """Show configuration dialog"""
        self.running = False
        dialog = ConfigDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.pop_size, self.mutation_rate, self.lifetime = dialog.result
            self.reset_simulation()
    
    def reset_simulation(self):
        """Reset the simulation"""
        if not self.pop_size:
            return
        
        self.running = False
        self.life_counter = 0
        self.best_fitness_ever = 0.0
        self.record_time = self.lifetime  # Initialize to lifetime (will be beaten if any rocket reaches target)
        self.population = Population(
            self.mutation_rate,
            self.pop_size,
            self.lifetime,
            self.start_pos[:],
            self.obstacles,
            self.target
        )
        self.start_button.config(text="START", bg="#00ff88")
        self.update_display()
    
    def toggle_simulation(self):
        """Start or pause simulation"""
        if not self.population:
            self.show_config()
            return
        
        self.running = not self.running
        if self.running:
            self.start_button.config(text="PAUSE", bg="#ffd700")
            self.simulate()
        else:
            self.start_button.config(text="RESUME", bg="#00ff88")
    
    def move_target(self, event):
        """Move target to clicked position"""
        # Update target position
        new_x = event.x - 12  # Center on click (24 width = ±12)
        new_y = event.y - 12
        
        self.target.position = [new_x, new_y]
        self.target_pos = [event.x, event.y]
        
        # Reset record time when target moves
        self.record_time = self.lifetime
    
    def simulate(self):
        """
        Run simulation loop.
        
        GENETIC ALGORITHM CYCLE:
        1. Live: Rockets execute behavior for lifetime frames
        2. Fitness: Calculate how well each rocket performed
        3. Selection: Build mating pool based on fitness
        4. Reproduction: Create new generation
        5. Repeat
        """
        if not self.running:
            return
        
        # If generation is still alive
        if self.life_counter < self.lifetime:
            self.population.live()
            
            # Check if any rocket reached target (for record tracking)
            if self.population.target_reached() and self.life_counter < self.record_time:
                self.record_time = self.life_counter
            
            self.life_counter += 1
        else:
            # Generation ended - create new generation
            self.life_counter = 0
            
            # Calculate fitness for all rockets
            self.population.calculate_fitness()
            
            # Find the best fitness in this generation
            generation_best = self.population.get_max_fitness()
            
            # Update all-time best if this generation did better
            if generation_best > self.best_fitness_ever:
                self.best_fitness_ever = generation_best
            
            # Update record time: find best finish time among rockets that hit target
            best = min((r.finish_time for r in self.population.population if r.hit_target), default=None)
            if best is not None:
                if self.record_time == self.lifetime or best < self.record_time:
                    self.record_time = best
            
            self.record_label.config(text=f"Record Time: {self.record_time}")
            
            # Continue genetic algorithm
            self.population.selection()
            self.population.reproduction()
        
        self.update_display()
        self.root.after(self.speed, self.simulate)
    
    def update_display(self):
        """Update visual display"""
        self.canvas.delete("all")
        
        # Draw obstacles
        for obs in self.obstacles:
            self.canvas.create_rectangle(
                obs.position[0], obs.position[1],
                obs.position[0] + obs.w, obs.position[1] + obs.h,
                fill="#555555",
                outline="#666666",
                width=2
            )
        
        # Draw target
        self.canvas.create_oval(
            self.target.position[0], self.target.position[1],
            self.target.position[0] + self.target.w, self.target.position[1] + self.target.h,
            fill="#ff6b6b",
            outline="#ffffff",
            width=2
        )
        
        # Draw start position
        self.canvas.create_oval(
            self.start_pos[0] - 8, self.start_pos[1] - 8,
            self.start_pos[0] + 8, self.start_pos[1] + 8,
            fill="#00ff88",
            outline="#ffffff",
            width=2
        )
        
        # Find best rocket in current frame (for highlighting during live phase)
        best_rocket = None
        current_max = 0
        if self.population:
            for rocket in self.population.population:
                if rocket.fitness > current_max:
                    current_max = rocket.fitness
                    best_rocket = rocket
        
        # Draw all rockets
        if self.population:
            for rocket in self.population.population:
                x, y = rocket.position
                r = rocket.r
                
                # Calculate heading angle from velocity
                if rocket.velocity[0] != 0 or rocket.velocity[1] != 0:
                    angle = math.atan2(rocket.velocity[1], rocket.velocity[0])
                else:
                    angle = 0
                
                # Rotate angle for proper orientation (up is default)
                angle += math.pi / 2
                
                # Draw rocket as triangle
                # Tip
                tip_x = x + r * 2 * math.cos(angle - math.pi/2)
                tip_y = y + r * 2 * math.sin(angle - math.pi/2)
                
                # Left corner
                left_x = x + r * math.cos(angle + math.pi/2 + 2.5)
                left_y = y + r * math.sin(angle + math.pi/2 + 2.5)
                
                # Right corner
                right_x = x + r * math.cos(angle + math.pi/2 - 2.5)
                right_y = y + r * math.sin(angle + math.pi/2 - 2.5)
                
                # Color based on status
                if rocket == best_rocket:
                    color = "#00d4ff"  # Bright cyan for best
                    outline_color = "#ffffff"
                    width = 2
                elif rocket.hit_target:
                    color = "#00ff88"  # Green for hit target
                    outline_color = "#ffffff"
                    width = 1
                elif rocket.hit_obstacle:
                    continue  # Don't draw crashed rockets
                else:
                    color = "#aaaaaa"  # Gray for others
                    outline_color = "#666666"
                    width = 1
                
                self.canvas.create_polygon(
                    tip_x, tip_y,
                    left_x, left_y,
                    right_x, right_y,
                    fill=color,
                    outline=outline_color,
                    width=width
                )
        
        # Update stats
        if self.population:
            self.gen_label.config(text=f"Generation: {self.population.get_generations()}")
            self.life_label.config(text=f"Frames Left: {self.lifetime - self.life_counter}")
            
            # Display fitness in scientific notation to handle very small/large numbers
            if self.best_fitness_ever > 0:
                self.fitness_label.config(text=f"Max Fitness: {self.best_fitness_ever:.3e}")
            else:
                self.fitness_label.config(text="Max Fitness: 0.00")
            
            # Display record time
            if self.record_time < self.lifetime:
                self.record_label.config(text=f"Record Time: {self.record_time:.3e}")
            else:
                self.record_label.config(text="Record Time: --")


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartRocketsGUI(root)
    root.mainloop()