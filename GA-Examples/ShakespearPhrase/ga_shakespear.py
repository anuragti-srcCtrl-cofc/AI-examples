import tkinter as tk
from tkinter import ttk, messagebox
import random
import string

"""
GENETIC ALGORITHM - EVOLVING SHAKESPEARE
=========================================

This program demonstrates a genetic algorithm that evolves random strings
into a target phrase ("To be or not to be.") using principles of natural evolution:

KEY CONCEPTS:
-------------
- POPULATION: A group of candidate solutions (random strings)
- FITNESS: How close each candidate is to the target
- SELECTION: Choosing the best candidates to reproduce
- CROSSOVER: Combining two parents to create offspring
- MUTATION: Random changes to introduce diversity

HOW IT WORKS:
-------------
1. Start with random strings (Generation 0)
2. Evaluate each string's fitness (% of correct characters)
3. Select the best performers for reproduction (natural selection)
4. Create new generation through crossover and mutation
5. Repeat steps 2-4 until we evolve the target phrase

This mimics biological evolution where:
- Organisms with beneficial traits survive and reproduce
- Traits are inherited and mixed through reproduction
- Random mutations introduce new possibilities
- Over many generations, populations adapt to their environment

The algorithm typically finds the solution in 100-500 generations!
"""

class DNA:
    """
    Represents a genetic sequence (genotype) for our virtual organism.
    
    In this genetic algorithm, each DNA object represents a potential solution
    to our problem (matching the target phrase). The 'genes' are individual
    characters that make up the phrase.
    """
    
    def __init__(self, length, genes=None):
        """
        Constructor: Creates DNA sequence (random or from provided genes).
        
        Args:
            length: Number of genes (characters) in the DNA sequence
            genes: Optional list of characters to use instead of random generation
            
        The genes are initialized randomly from printable ASCII characters (space through ~).
        This randomness is crucial - we start with completely random "guesses" and let
        evolution find the solution through selection and reproduction.
        """
        if genes:
            # Use provided genes (for user-specified starting population)
            self.genes = list(genes)
        else:
            # Generate random genes: each gene is a random printable character
            # string.printable[:95] gives us characters from space (32) to ~ (126)
            self.genes = [random.choice(string.printable[:95]) for _ in range(length)]
        
        # Fitness score: measures how close this DNA is to the target (0.0 to 1.0)
        self.fitness = 0.0
    
    def get_phrase(self):
        """
        Converts the gene array into a readable string.
        
        Returns:
            String representation of the DNA (the actual phrase)
        """
        return ''.join(self.genes)
    
    def calculate_fitness(self, target):
        """
        FITNESS FUNCTION - Core of natural selection!
        
        Calculates how "fit" this DNA is by comparing it to the target phrase.
        Fitness is the percentage of characters that match the target in the correct position.
        
        Args:
            target: The target phrase we're trying to evolve toward
            
        A fitness of 1.0 means perfect match (we've evolved the target phrase).
        A fitness of 0.0 means no characters match.
        
        This fitness score determines reproductive success - higher fitness means
        more likely to be selected as a parent for the next generation.
        """
        # Count how many characters match the target at the same position
        score = sum(1 for i in range(len(self.genes)) if self.genes[i] == target[i])
        
        # Normalize to a value between 0.0 and 1.0
        self.fitness = score / len(target)
    
    def crossover(self, partner):
        """
        CROSSOVER (REPRODUCTION) - Creates offspring from two parents.
        
        This simulates sexual reproduction where a child inherits traits from both parents.
        We use "single-point crossover": pick a random midpoint, take genes from one parent
        before that point and genes from the other parent after that point.
        
        Args:
            partner: The other parent DNA to mate with
            
        Returns:
            A new DNA object (child) with genes from both parents
            
        Example:
            Parent A: "Hello"
            Parent B: "World"
            Midpoint: 2
            Child:    "Herld" (He from B, rld from A)
        """
        # Create a new child DNA with the same length as parents
        child = DNA(len(self.genes))
        
        # Pick a random crossover point
        midpoint = random.randint(0, len(self.genes) - 1)
        
        # Inherit genes from both parents based on the midpoint
        # Before midpoint: inherit from partner
        # After midpoint: inherit from self
        for i in range(len(self.genes)):
            child.genes[i] = self.genes[i] if i > midpoint else partner.genes[i]
        
        return child
    
    def mutate(self, mutation_rate):
        """
        MUTATION - Randomly alters genes to introduce genetic diversity.
        
        Mutation is crucial! Without it, we can only recombine existing genes.
        Mutation introduces new genetic material into the population, allowing
        us to explore solutions that wouldn't be possible through crossover alone.
        
        Args:
            mutation_rate: Probability (0.0 to 1.0) that each gene will mutate
                          Typical values: 0.01 (1%) - 0.05 (5%)
                          
        For each gene, we roll the dice. If random() < mutation_rate, that gene
        gets replaced with a completely random character.
        
        Think of this like copying errors when DNA replicates in nature.
        """
        for i in range(len(self.genes)):
            # Roll the dice for each gene
            if random.random() < mutation_rate:
                # This gene mutates! Replace it with a random character
                self.genes[i] = random.choice(string.printable[:95])


class Population:
    """
    Manages the entire population of DNA organisms and implements the genetic algorithm.
    
    This class orchestrates the evolutionary process:
    1. Maintains a population of DNA objects (potential solutions)
    2. Evaluates fitness of each member
    3. Performs natural selection (survival of the fittest)
    4. Creates new generations through reproduction
    
    The algorithm repeats these steps until we evolve the target phrase.
    """
    
    def __init__(self, target, mutation_rate, pop_size, seed_phrases=None):
        """
        Initialize a new population.
        
        Args:
            target: The phrase we're trying to evolve toward
            mutation_rate: Probability of mutation (0.01 = 1% chance per gene)
            pop_size: Number of organisms in the population (e.g., 150)
            seed_phrases: Optional list of phrases to seed the initial population
            
        We start with a population of random DNA sequences, but can optionally
        include user-provided seed phrases that might give evolution a head start.
        """
        self.target = target
        self.mutation_rate = mutation_rate
        
        # Create initial population
        self.population = []
        
        # Add seed phrases if provided (user-specified starting organisms)
        if seed_phrases:
            for phrase in seed_phrases:
                self.population.append(DNA(len(target), genes=phrase))
        
        # Fill remaining population with random DNA
        remaining = pop_size - len(self.population)
        for _ in range(remaining):
            self.population.append(DNA(len(target)))
        
        # Mating pool: organisms that can reproduce (selected based on fitness)
        # Think of this as "who gets to have children"
        self.mating_pool = []
        
        # Track number of generations (iterations of the algorithm)
        self.generations = 0
        
        # Flag to stop when we've found the target
        self.finished = False
        
        # Calculate initial fitness for all organisms
        self.calculate_fitness()
    
    def calculate_fitness(self):
        """
        Evaluate how fit each organism is by comparing to the target.
        
        This is Step 1 of each generation: EVALUATION
        
        We need to know how good each solution is before we can select
        the best ones for reproduction.
        """
        for dna in self.population:
            dna.calculate_fitness(self.target)
    
    def natural_selection(self):
        """
        NATURAL SELECTION - "Survival of the fittest"
        
        Creates a mating pool where fitter organisms appear more frequently.
        This implements the core principle of evolution: organisms with traits
        better suited to their environment are more likely to reproduce.
        
        HOW IT WORKS:
        1. Find the organism with the highest fitness
        2. Normalize all fitness values relative to the best (0.0 to 1.0)
        3. Add each organism to the mating pool multiple times based on fitness
           - High fitness = many copies = more likely to be selected as parent
           - Low fitness = few/no copies = less likely to reproduce
           
        Example:
            If organism A has 100% fitness and organism B has 50% fitness:
            - A gets added ~100 times to the mating pool
            - B gets added ~50 times to the mating pool
            - A is twice as likely to be selected for reproduction
            
        This is FITNESS-PROPORTIONATE SELECTION (also called "roulette wheel selection").
        """
        # Clear the mating pool from the previous generation
        self.mating_pool.clear()

        # Find the maximum fitness in the current population
        # This will be our reference point for normalization
        max_fitness = max(dna.fitness for dna in self.population)
        
        # Edge case: if everyone has 0 fitness, avoid division by zero
        if max_fitness == 0:
            max_fitness = 1
        
        # Build the mating pool
        for dna in self.population:
            # Normalize fitness to 0-1 range based on the best organism
            normalized_fitness = dna.fitness / max_fitness
            
            # Convert to number of copies in mating pool
            # Multiply by 100 as an arbitrary scaling factor
            # This means a perfect organism (fitness=1.0) gets 100 copies
            n = int(normalized_fitness * 100)
            
            # Add this organism 'n' times to the mating pool
            # More copies = higher probability of being selected for reproduction
            self.mating_pool.extend([dna] * n)
    
    def generate(self):
        """
        REPRODUCTION - Create a new generation of organisms.
        
        This is the heart of the genetic algorithm! We create an entirely new
        population by selecting parents from the mating pool and producing offspring.
        
        PROCESS for each new organism:
        1. Pick two random parents from the mating pool
           (fitter parents are more likely to be picked - they appear more often)
        2. CROSSOVER: Create a child by combining parent genes
        3. MUTATION: Randomly alter some of the child's genes
        4. Add child to the new population
        
        After this, the old generation is completely replaced by the new generation.
        This is how evolution works - each generation is (hopefully) slightly better
        than the last.
        """
        # Safety check: make sure we have a mating pool
        if not self.mating_pool:
            return
        
        # Create the new generation
        new_population = []
        
        # Create exactly as many children as the previous population size
        for _ in range(len(self.population)):
            # SELECT: Pick two random parents from the mating pool
            # Because fitter organisms appear more in the pool, they're more likely picked
            parent_a = random.choice(self.mating_pool)
            parent_b = random.choice(self.mating_pool)
            
            # CROSSOVER: Combine parent genes to create a child
            child = parent_a.crossover(parent_b)
            
            # MUTATION: Randomly alter some genes (introduces genetic diversity)
            child.mutate(self.mutation_rate)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Replace the old population with the new generation
        # The old generation is gone - only the offspring survive
        self.population = new_population
        
        # Increment generation counter
        self.generations += 1
    
    def get_best(self):
        """
        Find the most fit organism in the current population.
        
        Returns:
            The DNA object with the highest fitness score
            
        This is our "best guess" at the current generation.
        We also check if we've found a perfect match (fitness = 1.0).
        """
        # Find organism with maximum fitness
        best = max(self.population, key=lambda dna: dna.fitness)
        
        # Check if we've achieved perfection!
        if best.fitness >= 1.0:
            self.finished = True
            
        return best
    
    def get_average_fitness(self):
        """
        Calculate the average fitness across the entire population.
        
        Returns:
            Mean fitness value (0.0 to 1.0)
            
        This metric helps us understand how the population is improving over time.
        We expect this to gradually increase as evolution progresses.
        """
        return sum(dna.fitness for dna in self.population) / len(self.population)
    
    def get_top_phrases(self, n=10):
        """
        Get the N best phrases from the current population.
        
        Args:
            n: Number of top phrases to return
            
        Returns:
            List of the top N phrase strings, sorted by fitness
            
        Useful for visualizing the diversity in the population and seeing
        multiple "good" solutions, not just the best one.
        """
        # Sort population by fitness (highest first)
        sorted_pop = sorted(self.population, key=lambda dna: dna.fitness, reverse=True)
        
        # Return the phrases (not the DNA objects) of the top N
        return [dna.get_phrase() for dna in sorted_pop[:n]]


class ConfigDialog:
    """Dialog for configuring target phrase and seed phrases"""
    
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configuration")
        self.dialog.geometry("600x650")
        self.dialog.configure(bg="#1a1a2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Title
        title = tk.Label(
            self.dialog,
            text="Configuration",
            font=("Arial", 18, "bold"),
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title.pack(pady=20)
        
        # Target phrase section
        target_frame = tk.Frame(self.dialog, bg="#16213e", relief=tk.RAISED, bd=2)
        target_frame.pack(padx=20, pady=10, fill=tk.X)
        
        tk.Label(
            target_frame,
            text="Target Phrase:",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        self.target_entry = tk.Entry(
            target_frame,
            font=("Courier", 12),
            bg="#0f0f1e",
            fg="#ffffff",
            insertbackground="#00d4ff"
        )
        self.target_entry.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.target_entry.insert(0, "To be or not to be.")
        
        # Parameters section
        params_frame = tk.Frame(self.dialog, bg="#16213e", relief=tk.RAISED, bd=2)
        params_frame.pack(padx=20, pady=10, fill=tk.X)
        
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
        self.pop_entry.insert(0, "150")
        
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
        
        # Seed phrases section
        seed_frame = tk.Frame(self.dialog, bg="#16213e", relief=tk.RAISED, bd=2)
        seed_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(
            seed_frame,
            text="Seed Phrases (optional, one per line):",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        tk.Label(
            seed_frame,
            text="Must be same length as target phrase",
            font=("Arial", 9, "italic"),
            bg="#16213e",
            fg="#888"
        ).pack(anchor=tk.W, padx=10)
        
        self.seed_text = tk.Text(
            seed_frame,
            font=("Courier", 10),
            bg="#0f0f1e",
            fg="#00ff88",
            insertbackground="#00d4ff",
            height=8
        )
        self.seed_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg="#1a1a2e")
        button_frame.pack(pady=20)
        
        tk.Button(
            button_frame,
            text="Start Evolution",
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
        target = self.target_entry.get().strip()
        if not target:
            messagebox.showerror("Error", "Target phrase cannot be empty!")
            return
        
        # Validate population size
        try:
            pop_size = int(self.pop_entry.get().strip())
            if pop_size < 10:
                messagebox.showerror("Error", "Population size must be at least 10!")
                return
        except ValueError:
            messagebox.showerror("Error", "Population size must be a valid number!")
            return
        
        # Validate mutation rate
        try:
            mutation_pct = float(self.mutation_entry.get().strip())
            if mutation_pct < 0 or mutation_pct > 100:
                messagebox.showerror("Error", "Mutation rate must be between 0 and 100!")
                return
            mutation_rate = mutation_pct / 100.0
        except ValueError:
            messagebox.showerror("Error", "Mutation rate must be a valid number!")
            return
        
        seed_phrases = []
        seed_text = self.seed_text.get("1.0", tk.END).strip()
        
        if seed_text:
            lines = [line.strip() for line in seed_text.split('\n') if line.strip()]
            for line in lines:
                if len(line) != len(target):
                    messagebox.showerror(
                        "Error", 
                        f"Seed phrase '{line}' length ({len(line)}) doesn't match target length ({len(target)})"
                    )
                    return
                seed_phrases.append(line)
        
        self.result = (target, pop_size, mutation_rate, seed_phrases)
        self.dialog.destroy()
    
    def cancel(self):
        self.dialog.destroy()


class GeneticAlgorithmGUI:
    """Beautiful GUI for genetic algorithm visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm - Evolving Shakespeare")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")
        
        # Algorithm parameters
        self.target = None
        self.seed_phrases = None
        self.pop_size = 150
        self.mutation_rate = 0.01
        self.population = None
        self.running = False
        self.speed = 50  # ms between generations
        
        self.setup_ui()
        self.show_config()
    
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title = tk.Label(
            self.root,
            text="GENETIC ALGORITHM",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title.pack(pady=20)
        
        # Main display frame
        main_frame = tk.Frame(self.root, bg="#16213e", relief=tk.RAISED, bd=2)
        main_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Target phrase
        target_label = tk.Label(
            main_frame,
            text="TARGET:",
            font=("Courier", 12, "bold"),
            bg="#16213e",
            fg="#888"
        )
        target_label.pack(pady=(20, 5))
        
        self.target_display = tk.Label(
            main_frame,
            text="",
            font=("Courier", 28, "bold"),
            bg="#16213e",
            fg="#ffffff"
        )
        self.target_display.pack(pady=5)
        
        # Best phrase
        best_label = tk.Label(
            main_frame,
            text="BEST MATCH:",
            font=("Courier", 12, "bold"),
            bg="#16213e",
            fg="#888"
        )
        best_label.pack(pady=(30, 5))
        
        self.best_display = tk.Label(
            main_frame,
            text="",
            font=("Courier", 28, "bold"),
            bg="#16213e",
            fg="#00ff88",
            height=2
        )
        self.best_display.pack(pady=5)
        
        # Fitness bar
        fitness_frame = tk.Frame(main_frame, bg="#16213e")
        fitness_frame.pack(pady=20, padx=40, fill=tk.X)
        
        tk.Label(
            fitness_frame,
            text="FITNESS:",
            font=("Arial", 11, "bold"),
            bg="#16213e",
            fg="#888"
        ).pack(anchor=tk.W)
        
        self.fitness_canvas = tk.Canvas(
            fitness_frame,
            height=30,
            bg="#0f0f1e",
            highlightthickness=0
        )
        self.fitness_canvas.pack(fill=tk.X, pady=5)
        
        # Stats frame
        stats_frame = tk.Frame(main_frame, bg="#16213e")
        stats_frame.pack(pady=10, padx=40, fill=tk.X)
        
        self.gen_label = tk.Label(
            stats_frame,
            text="Generation: 0",
            font=("Arial", 11),
            bg="#16213e",
            fg="#ffffff"
        )
        self.gen_label.pack(anchor=tk.W, pady=2)
        
        self.avg_fitness_label = tk.Label(
            stats_frame,
            text="Average Fitness: 0.00%",
            font=("Arial", 11),
            bg="#16213e",
            fg="#ffffff"
        )
        self.avg_fitness_label.pack(anchor=tk.W, pady=2)
        
        self.pop_label = tk.Label(
            stats_frame,
            text=f"Population: {self.pop_size}",
            font=("Arial", 11),
            bg="#16213e",
            fg="#ffffff"
        )
        self.pop_label.pack(anchor=tk.W, pady=2)
        
        self.mutation_label = tk.Label(
            stats_frame,
            text=f"Mutation Rate: {int(self.mutation_rate * 100)}%",
            font=("Arial", 11),
            bg="#16213e",
            fg="#ffffff"
        )
        self.mutation_label.pack(anchor=tk.W, pady=2)
        
        # Top phrases display
        phrases_label = tk.Label(
            main_frame,
            text="TOP CANDIDATES:",
            font=("Courier", 10, "bold"),
            bg="#16213e",
            fg="#888"
        )
        phrases_label.pack(pady=(10, 5))
        
        self.phrases_text = tk.Text(
            main_frame,
            height=6,
            width=40,
            font=("Courier", 9),
            bg="#0f0f1e",
            fg="#00d4ff",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.phrases_text.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="#1a1a2e")
        button_frame.pack(pady=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="START",
            command=self.toggle_evolution,
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
        
        reset_button = tk.Button(
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
        )
        reset_button.pack(side=tk.LEFT, padx=10)
    
    def show_config(self):
        """Show configuration dialog"""
        self.running = False
        dialog = ConfigDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.target, self.pop_size, self.mutation_rate, self.seed_phrases = dialog.result
            self.reset_population()
    
    def reset_population(self):
        """Reset the population"""
        if not self.target:
            return
            
        self.running = False
        self.population = Population(self.target, self.mutation_rate, self.pop_size, self.seed_phrases)
        self.start_button.config(text="START", bg="#00ff88")
        self.target_display.config(text=self.target)
        self.pop_label.config(text=f"Population: {self.pop_size}")
        self.mutation_label.config(text=f"Mutation Rate: {int(self.mutation_rate * 100)}%")
        self.update_display()
    
    def toggle_evolution(self):
        """Start or pause evolution"""
        if not self.population:
            self.show_config()
            return
            
        self.running = not self.running
        if self.running:
            self.start_button.config(text="PAUSE", bg="#ffd700")
            self.evolve()
        else:
            self.start_button.config(text="RESUME", bg="#00ff88")
    
    def evolve(self):
        """
        Run one complete generation of the genetic algorithm.
        
        THE GENETIC ALGORITHM CYCLE:
        =============================
        1. SELECTION: Create mating pool based on fitness (natural_selection)
        2. REPRODUCTION: Create new generation from mating pool (generate)
        3. EVALUATION: Calculate fitness of new generation (calculate_fitness)
        4. REPEAT until target is found or we stop manually
        
        This method is called repeatedly (via tkinter's after() method) to
        simulate continuous evolution. Each call represents one generation.
        """
        # Stop conditions
        if not self.running or self.population.finished:
            if self.population.finished:
                self.start_button.config(text="COMPLETE", bg="#00cc6a")
            return
        
        # STEP 1: Natural Selection - select organisms for reproduction
        # Creates mating pool where fitter organisms are more represented
        self.population.natural_selection()
        
        # STEP 2: Reproduction - create the next generation
        # Picks parents from mating pool, performs crossover and mutation
        self.population.generate()
        
        # STEP 3: Evaluation - calculate fitness of the new generation
        # Determines how good each new organism is
        self.population.calculate_fitness()
        
        # Update the visual display with new generation's data
        self.update_display()
        
        # Schedule the next generation
        # This creates a loop: evolve() schedules itself to run again
        self.root.after(self.speed, self.evolve)
    
    def update_display(self):
        """Update all display elements"""
        best = self.population.get_best()
        best_phrase = best.get_phrase()
        
        # Update best phrase with color coding
        self.best_display.config(text=best_phrase)
        
        # Update fitness bar
        self.fitness_canvas.delete("all")
        width = self.fitness_canvas.winfo_width()
        if width <= 1:
            width = 600
        
        bar_width = int(width * best.fitness)
        
        # Color gradient based on fitness
        if best.fitness < 0.33:
            color = "#ff6b6b"
        elif best.fitness < 0.66:
            color = "#ffd700"
        else:
            color = "#00ff88"
        
        self.fitness_canvas.create_rectangle(
            0, 0, bar_width, 30,
            fill=color,
            outline=""
        )
        
        # Fitness percentage text
        fitness_pct = f"{best.fitness * 100:.1f}%"
        self.fitness_canvas.create_text(
            width // 2, 15,
            text=fitness_pct,
            font=("Arial", 12, "bold"),
            fill="#ffffff"
        )
        
        # Update stats
        self.gen_label.config(text=f"Generation: {self.population.generations}")
        avg_fitness = self.population.get_average_fitness()
        self.avg_fitness_label.config(text=f"Average Fitness: {avg_fitness * 100:.2f}%")
        
        # Update top phrases
        self.phrases_text.delete(1.0, tk.END)
        top_phrases = self.population.get_top_phrases(6)
        for i, phrase in enumerate(top_phrases, 1):
            self.phrases_text.insert(tk.END, f"{i}. {phrase}\n")
        
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()