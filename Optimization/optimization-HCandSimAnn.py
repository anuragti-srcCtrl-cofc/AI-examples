"""
Optimization 270: Charging Station Placement Visualizer
======================================================
Places 3 EV charging stations on a 20x8 grid to minimize total Manhattan distance
to 9 randomly located houses.

Supported Algorithms:
1. Hill Climbing (Steepest Descent)
   - Optional: Random Restart (5-20 restarts)
   - Optional: Local Beam Search (beam width 2-5)
2. Simulated Annealing
   - Configurable Initial Max Temperature (recommended: 100.0)
   - Configurable Cooling Rate (recommended: 0.95)

Features:
- GUI built with Tkinter + embedded Matplotlib
- Black background with crisp white grid lines
- Houses shown in bright cyan (⌂), charging stations in bright yellow (⚡)
- Interactive 'Step' (1 step at a time) and 'Run / Pause' controls
- 'Reset Stations' preserves house positions and randomizes charging stations
"""

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =====================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# =====================================================================
GRID_COLS = 20
GRID_ROWS = 8
NUM_HOUSES = 9
NUM_STATIONS = 3

# Unicode glyphs and styling
HOUSE_SYMBOL = '\u2302'       # ⌂ Unicode House
STATION_SYMBOL = '\u26A1'     # ⚡ Unicode Lightning / Charging
HOUSE_COLOR = '#00FFFF'       # Bright Cyan
STATION_COLOR = '#FFD700'     # Bright Gold / Yellow
HOUSE_FONTSIZE = 42           # Visually prominent house glyph size
STATION_FONTSIZE = 38         # Visually prominent charging station size
GRID_LINE_COLOR = '#FFFFFF'   # White grid lines
BACKGROUND_COLOR = '#000000'  # Solid Black


# =====================================================================
# SECTION 2: PROBLEM MODEL & DISTANCE CALCULATIONS
# =====================================================================
def manhattan_distance(p1, p2):
    """Calculates Manhattan (L1) distance between two (x, y) coordinates."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def total_manhattan_cost(stations, houses):
    """
    Computes objective cost: Sum of Manhattan distances from each house
    to its closest charging station.
    """
    total_cost = 0
    for h in houses:
        min_dist = min(manhattan_distance(h, s) for s in stations)
        total_cost += min_dist
    return total_cost


def get_station_neighbors(stations, cols=GRID_COLS, rows=GRID_ROWS):
    """
    Generates all valid immediate 1-step orthogonal neighbors for the charging stations.
    At each step, one station can move Up, Down, Left, or Right by 1 cell.
    """
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    station_list = list(stations)

    for i in range(len(station_list)):
        sx, sy = station_list[i]
        for dx, dy in moves:
            nx, ny = sx + dx, sy + dy
            # Check bounds and avoid placing two stations on the same cell
            if 0 <= nx < cols and 0 <= ny < rows:
                candidate = list(station_list)
                candidate[i] = (nx, ny)
                # Keep stations sorted for canonical representation
                candidate_tuple = tuple(sorted(candidate))
                if candidate_tuple != stations and candidate_tuple not in neighbors:
                    neighbors.append(candidate_tuple)
    return neighbors


def random_station_positions(cols=GRID_COLS, rows=GRID_ROWS, count=NUM_STATIONS, avoid=None):
    """Generates `count` unique random station positions."""
    avoid_set = set(avoid) if avoid else set()
    available = [(x, y) for x in range(cols) for y in range(rows) if (x, y) not in avoid_set]
    return tuple(sorted(random.sample(available, count)))


# =====================================================================
# SECTION 3: OPTIMIZATION ALGORITHM GENERATORS (STEP-BY-STEP)
# =====================================================================
# Each algorithm is written as a Python generator.
# On each yield, it provides:
# (current_stations, current_cost, status_text, is_finished)

def hill_climbing_generator(initial_stations, houses):
    """
    Standard Steepest-Descent Hill Climbing.
    Evaluates all 1-step neighbors, moves to the best neighbor if it improves cost.
    Stops when no neighbor offers a lower cost (local minimum).
    """
    current = initial_stations
    current_cost = total_manhattan_cost(current, houses)
    step = 0

    yield current, current_cost, f"Initial state. Cost: {current_cost}", False

    while True:
        step += 1
        neighbors = get_station_neighbors(current)
        if not neighbors:
            break

        # Find neighbor with lowest cost
        best_neighbor = None
        best_cost = current_cost

        for n in neighbors:
            c = total_manhattan_cost(n, houses)
            if c < best_cost:
                best_cost = c
                best_neighbor = n

        # If an improving neighbor is found, make the move
        if best_neighbor is not None:
            old_cost = current_cost
            current = best_neighbor
            current_cost = best_cost
            msg = f"Step {step}: Moved to better neighbor. Cost: {old_cost} -> {current_cost}"
            yield current, current_cost, msg, False
        else:
            # Reached a local optimum
            msg = f"Local minimum reached at step {step}. Best Cost: {current_cost}"
            yield current, current_cost, msg, True
            break


def random_restart_hill_climbing_generator(initial_stations, houses, restarts=10, beam_width=1):
    """
    Random-Restart Hill Climbing (supports both standard and Local Beam Search).
    Continuously tracks the best solution found so far across all restarts.
    At the conclusion of all restarts, automatically displays the best overall solution.
    """
    best_overall_stations = initial_stations
    best_overall_cost = total_manhattan_cost(initial_stations, houses)

    for r in range(1, restarts + 1):
        if r == 1:
            start_stations = initial_stations
        else:
            start_stations = random_station_positions(avoid=houses)

        # Initialize beam or single state
        if beam_width > 1:
            beam = [start_stations]
            while len(beam) < beam_width:
                rnd = random_station_positions(avoid=houses)
                if rnd not in beam:
                    beam.append(rnd)
            best_state = min(beam, key=lambda s: total_manhattan_cost(s, houses))
            best_cost = total_manhattan_cost(best_state, houses)
            mode_name = f"Beam Search (w={beam_width})"
        else:
            best_state = start_stations
            best_cost = total_manhattan_cost(best_state, houses)
            mode_name = "Hill Climbing"

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_stations = best_state

        msg = f"[Restart {r}/{restarts}] Starting new {mode_name}. Cost: {best_cost} (Best Ever: {best_overall_cost})"
        yield best_state, best_cost, msg, False, best_overall_cost

        step = 0
        stagnant_steps = 0

        while True:
            step += 1
            if beam_width > 1:
                # Beam search step
                all_candidates = set()
                for state in beam:
                    for n in get_station_neighbors(state):
                        all_candidates.add(n)

                if not all_candidates:
                    break

                scored = [(n, total_manhattan_cost(n, houses)) for n in all_candidates]
                scored.sort(key=lambda x: x[1])

                new_beam = [s for s, _ in scored[:beam_width]]
                cand_state = scored[0][0]
                cand_cost = scored[0][1]

                if cand_cost < best_cost:
                    old_cost = best_cost
                    best_cost = cand_cost
                    best_state = cand_state
                    beam = new_beam
                    stagnant_steps = 0

                    if best_cost < best_overall_cost:
                        best_overall_cost = best_cost
                        best_overall_stations = best_state

                    msg = f"[Restart {r}/{restarts}] Step {step}: Cost: {old_cost} -> {best_cost} (Best Ever: {best_overall_cost})"
                    yield best_state, best_cost, msg, False, best_overall_cost
                else:
                    stagnant_steps += 1
                    beam = new_beam
                    if stagnant_steps >= 3:
                        msg = f"[Restart {r}/{restarts}] Converged at Cost: {best_cost} (Best Ever: {best_overall_cost})"
                        yield best_state, best_cost, msg, False, best_overall_cost
                        break
                    else:
                        msg = f"[Restart {r}/{restarts}] Step {step}: No improvement ({stagnant_steps}/3) (Best Ever: {best_overall_cost})"
                        yield best_state, best_cost, msg, False, best_overall_cost
            else:
                # Standard hill climbing step
                neighbors = get_station_neighbors(best_state)
                best_neighbor = None
                neighbor_best_cost = best_cost

                for n in neighbors:
                    c = total_manhattan_cost(n, houses)
                    if c < neighbor_best_cost:
                        neighbor_best_cost = c
                        best_neighbor = n

                if best_neighbor is not None:
                    old_cost = best_cost
                    best_state = best_neighbor
                    best_cost = neighbor_best_cost

                    if best_cost < best_overall_cost:
                        best_overall_cost = best_cost
                        best_overall_stations = best_state

                    msg = f"[Restart {r}/{restarts}] Step {step}: Cost: {old_cost} -> {best_cost} (Best Ever: {best_overall_cost})"
                    yield best_state, best_cost, msg, False, best_overall_cost
                else:
                    # Reached local minimum for this restart
                    msg = f"[Restart {r}/{restarts}] Local min at Cost: {best_cost} (Best Ever: {best_overall_cost})"
                    yield best_state, best_cost, msg, False, best_overall_cost
                    break

    # Completed all restarts: show the best solution found overall
    yield best_overall_stations, best_overall_cost, f"Completed all {restarts} restarts! Best Solution Cost: {best_overall_cost}", True, best_overall_cost


def local_beam_search_generator(initial_stations, houses, beam_width=3):
    """
    Local Beam Search:
    Maintains `beam_width` candidate station configurations simultaneously.
    At each step, generates all neighbors from all beam candidates,
    evaluates them, and selects the top `beam_width` best unique configurations.
    """
    beam = [initial_stations]
    while len(beam) < beam_width:
        rnd = random_station_positions(avoid=houses)
        if rnd not in beam:
            beam.append(rnd)

    step = 0
    best_state = min(beam, key=lambda s: total_manhattan_cost(s, houses))
    best_cost = total_manhattan_cost(best_state, houses)

    yield best_state, best_cost, f"Initial Beam (width={beam_width}). Cost: {best_cost}", False, best_cost

    stagnant_steps = 0
    max_stagnant = 3

    while True:
        step += 1
        all_candidates = set()
        for state in beam:
            for n in get_station_neighbors(state):
                all_candidates.add(n)

        if not all_candidates:
            break

        scored_candidates = [(n, total_manhattan_cost(n, houses)) for n in all_candidates]
        scored_candidates.sort(key=lambda x: x[1])

        new_beam = [s for s, _ in scored_candidates[:beam_width]]
        new_best_state = scored_candidates[0][0]
        new_best_cost = scored_candidates[0][1]

        if new_best_cost < best_cost:
            old_cost = best_cost
            best_cost = new_best_cost
            best_state = new_best_state
            beam = new_beam
            stagnant_steps = 0
            msg = f"Beam Step {step}: Cost improved {old_cost} -> {best_cost}"
            yield best_state, best_cost, msg, False, best_cost
        else:
            stagnant_steps += 1
            beam = new_beam
            if stagnant_steps >= max_stagnant:
                msg = f"Beam Search converged at step {step}. Final Best Cost: {best_cost}"
                yield best_state, best_cost, msg, True, best_cost
                break
            else:
                msg = f"Beam Step {step}: No improvement ({stagnant_steps}/{max_stagnant}). Best Cost: {best_cost}"
                yield best_state, best_cost, msg, False, best_cost



def simulated_annealing_generator(initial_stations, houses, max_temp=100.0, cooling_rate=0.95):
    """
    Simulated Annealing:
    Picks a random 1-step neighbor.
    If it improves cost (dE < 0), always accepts it.
    If it worsens cost (dE >= 0), accepts it with probability P = exp(-dE / T).
    Decreases temperature T by cooling_rate at each step.
    """
    current = initial_stations
    current_cost = total_manhattan_cost(current, houses)
    best_stations = current
    best_cost = current_cost

    temperature = float(max_temp)
    min_temp = 0.05
    step = 0

    yield current, current_cost, f"Initial State. Cost: {current_cost}, Temp: {temperature:.2f}", False

    while temperature > min_temp:
        step += 1
        neighbors = get_station_neighbors(current)
        if not neighbors:
            break

        # Select a random neighbor
        neighbor = random.choice(neighbors)
        neighbor_cost = total_manhattan_cost(neighbor, houses)
        delta_e = neighbor_cost - current_cost

        accepted = False
        if delta_e < 0:
            # Better state -> always accept
            current = neighbor
            current_cost = neighbor_cost
            accepted = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_stations = current
            msg = f"Step {step}: Accepted better move (Cost: {current_cost}, T: {temperature:.2f})"
        else:
            # Worse state -> accept with Boltzmann probability
            prob = math.exp(-delta_e / temperature)
            r = random.random()
            if r < prob:
                current = neighbor
                current_cost = neighbor_cost
                accepted = True
                msg = f"Step {step}: Accepted worse move (Cost: {current_cost}, P={prob:.2f} > {r:.2f}, T: {temperature:.2f})"
            else:
                msg = f"Step {step}: Rejected worse move (dE=+{delta_e}, P={prob:.2f} <= {r:.2f}, T: {temperature:.2f})"

        # Cool down
        temperature *= cooling_rate

        yield current, current_cost, msg, False

    # Finished cooling
    yield best_stations, best_cost, f"Simulated Annealing finished. Lowest Cost Found: {best_cost}", True


# =====================================================================
# SECTION 4: GUI & VISUALIZATION (TKINTER + MATPLOTLIB)
# =====================================================================
class ChargingStationVisualizerApp:
    """Main Application GUI managing controls, animation, and Matplotlib display."""

    def __init__(self, root):
        self.root = root
        self.root.title("EV Charging Station Optimizer (20x8 Grid)")
        self.root.geometry("1280x760")
        self.root.configure(bg="#1E1E1E")

        # State initialization
        # Fix 9 random houses
        self.all_cells = [(x, y) for x in range(GRID_COLS) for y in range(GRID_ROWS)]
        self.houses = random.sample(self.all_cells, NUM_HOUSES)

        # 3 Random charging stations (distinct from houses)
        self.stations = random_station_positions(avoid=self.houses)
        self.current_cost = total_manhattan_cost(self.stations, self.houses)
        self.best_ever_cost = self.current_cost

        # Algorithm generator and animation state
        self.algorithm_gen = None
        self.is_running = False
        self.animation_speed_ms = 120  # Delay between steps for legible viewing

        # Build UI and initial draw
        self._build_gui()
        self._on_algorithm_change()
        self._redraw_board("Ready. Press 'Step' or 'Run' to begin.")

    def _build_gui(self):
        """Constructs the Tkinter layout: Controls on top/sides and Plot in center."""
        # Top control frame
        controls_frame = tk.Frame(self.root, bg="#2D2D2D", padx=10, pady=8)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        # --- Row 1: Algorithm selection and Board controls ---
        r1 = tk.Frame(controls_frame, bg="#2D2D2D")
        r1.pack(fill=tk.X, pady=3)

        tk.Label(r1, text="Algorithm:", fg="#FFFFFF", bg="#2D2D2D", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(5, 5))
        self.algo_var = tk.StringVar(value="Hill Climbing")
        self.algo_dropdown = ttk.Combobox(
            r1,
            textvariable=self.algo_var,
            values=["Hill Climbing", "Simulated Annealing"],
            state="readonly",
            width=20,
            font=("Segoe UI", 10)
        )
        self.algo_dropdown.pack(side=tk.LEFT, padx=5)
        self.algo_dropdown.bind("<<ComboboxSelected>>", lambda e: self._on_algorithm_change())

        # Reset button (preserves houses, randomizes charging stations)
        self.btn_reset = tk.Button(
            r1,
            text="Reset Stations",
            command=self.reset_stations,
            bg="#555555",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
            relief=tk.RAISED
        )
        self.btn_reset.pack(side=tk.LEFT, padx=15)

        # Step and Run buttons
        self.btn_step = tk.Button(
            r1,
            text="Step",
            command=self.step_algorithm,
            bg="#007ACC",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=2
        )
        self.btn_step.pack(side=tk.LEFT, padx=5)

        self.btn_run = tk.Button(
            r1,
            text="Run",
            command=self.toggle_run,
            bg="#28A745",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=2
        )
        self.btn_run.pack(side=tk.LEFT, padx=5)

        # --- Row 2: Algorithm Parameters ---
        params_frame = tk.Frame(controls_frame, bg="#252526", padx=8, pady=6, relief=tk.GROOVE, bd=1)
        params_frame.pack(fill=tk.X, pady=5)

        # Hill climbing parameter group
        self.hc_frame = tk.Frame(params_frame, bg="#252526")
        self.hc_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Random restart checkbox and entry
        self.restart_var = tk.BooleanVar(value=False)
        self.chk_restart = tk.Checkbutton(
            self.hc_frame,
            text="Random Restart",
            variable=self.restart_var,
            command=self._on_hc_options_toggle,
            bg="#252526",
            fg="#FFFFFF",
            selectcolor="#333333",
            activebackground="#252526",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9)
        )
        self.chk_restart.pack(side=tk.LEFT, padx=3)

        self.lbl_restarts = tk.Label(self.hc_frame, text="Restarts (5-20):", fg="#CCCCCC", bg="#252526", font=("Segoe UI", 9))
        self.lbl_restarts.pack(side=tk.LEFT, padx=(6, 2))
        self.entry_restarts = tk.Entry(self.hc_frame, width=4, font=("Segoe UI", 9))
        self.entry_restarts.insert(0, "10")
        self.entry_restarts.pack(side=tk.LEFT, padx=3)

        # Local beam search checkbox and entry
        self.beam_var = tk.BooleanVar(value=False)
        self.chk_beam = tk.Checkbutton(
            self.hc_frame,
            text="Local Beam Search",
            variable=self.beam_var,
            command=self._on_hc_options_toggle,
            bg="#252526",
            fg="#FFFFFF",
            selectcolor="#333333",
            activebackground="#252526",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9)
        )
        self.chk_beam.pack(side=tk.LEFT, padx=(15, 3))

        self.lbl_beam = tk.Label(self.hc_frame, text="Neighbors (2-5):", fg="#CCCCCC", bg="#252526", font=("Segoe UI", 9))
        self.lbl_beam.pack(side=tk.LEFT, padx=(6, 2))
        self.entry_beam = tk.Entry(self.hc_frame, width=4, font=("Segoe UI", 9))
        self.entry_beam.insert(0, "3")
        self.entry_beam.pack(side=tk.LEFT, padx=3)

        # Separator between HC and SA
        ttk.Separator(params_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=15)

        # Simulated Annealing parameter group
        self.sa_frame = tk.Frame(params_frame, bg="#252526")
        self.sa_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.lbl_temp = tk.Label(self.sa_frame, text="Max Temp:", fg="#CCCCCC", bg="#252526", font=("Segoe UI", 9))
        self.lbl_temp.pack(side=tk.LEFT, padx=3)
        self.entry_temp = tk.Entry(self.sa_frame, width=6, font=("Segoe UI", 9))
        self.entry_temp.insert(0, "100.0")  # Recommended temperature
        self.entry_temp.pack(side=tk.LEFT, padx=(2, 10))

        self.lbl_cooling = tk.Label(self.sa_frame, text="Cooling Rate:", fg="#CCCCCC", bg="#252526", font=("Segoe UI", 9))
        self.lbl_cooling.pack(side=tk.LEFT, padx=3)
        self.entry_cooling = tk.Entry(self.sa_frame, width=5, font=("Segoe UI", 9))
        self.entry_cooling.insert(0, "0.95")  # Recommended cooling rate
        self.entry_cooling.pack(side=tk.LEFT, padx=2)

        # --- Status banner ---
        self.status_label = tk.Label(
            controls_frame,
            text="Total Manhattan Distance: -- | Ready",
            fg="#00FFCC",
            bg="#1E1E1E",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            padx=6,
            pady=3
        )
        self.status_label.pack(fill=tk.X, pady=(4, 0))

        # --- Matplotlib Canvas ---
        plot_container = tk.Frame(self.root, bg=BACKGROUND_COLOR)
        plot_container.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(15, 6), facecolor=BACKGROUND_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_algorithm_change(self):
        """
        Dynamically enables and disables GUI inputs based on selected algorithm:
        - Hill Climbing: enables HC options (Random Restart, Beam Search), disables SA inputs.
        - Simulated Annealing: enables SA inputs (Max Temp, Cooling Rate), disables HC options.
        """
        algo = self.algo_var.get()
        self.stop_run()
        self.algorithm_gen = None

        if algo == "Hill Climbing":
            # Enable Hill Climbing checkboxes
            self.chk_restart.config(state=tk.NORMAL)
            self.chk_beam.config(state=tk.NORMAL)
            self._on_hc_options_toggle()

            # Disable Simulated Annealing fields
            self.entry_temp.config(state=tk.DISABLED)
            self.entry_cooling.config(state=tk.DISABLED)
            self.lbl_temp.config(fg="#666666")
            self.lbl_cooling.config(fg="#666666")
        else:
            # Disable Hill Climbing checkboxes and entries
            self.chk_restart.config(state=tk.DISABLED)
            self.chk_beam.config(state=tk.DISABLED)
            self.entry_restarts.config(state=tk.DISABLED)
            self.entry_beam.config(state=tk.DISABLED)
            self.lbl_restarts.config(fg="#666666")
            self.lbl_beam.config(fg="#666666")

            # Enable Simulated Annealing fields
            self.entry_temp.config(state=tk.NORMAL)
            self.entry_cooling.config(state=tk.NORMAL)
            self.lbl_temp.config(fg="#FFFFFF")
            self.lbl_cooling.config(fg="#FFFFFF")

    def _on_hc_options_toggle(self):
        """Manages mutually exclusive or enabled state for Hill Climbing text inputs."""
        # Random restart entry
        if self.restart_var.get():
            self.entry_restarts.config(state=tk.NORMAL)
            self.lbl_restarts.config(fg="#FFFFFF")
        else:
            self.entry_restarts.config(state=tk.DISABLED)
            self.lbl_restarts.config(fg="#666666")

        # Local beam search entry
        if self.beam_var.get():
            self.entry_beam.config(state=tk.NORMAL)
            self.lbl_beam.config(fg="#FFFFFF")
        else:
            self.entry_beam.config(state=tk.DISABLED)
            self.lbl_beam.config(fg="#666666")

        self.algorithm_gen = None

    def reset_stations(self):
        """
        Resets charging stations to 3 new random locations while keeping
        the 9 houses in their exact same positions.
        """
        self.stop_run()
        self.stations = random_station_positions(avoid=self.houses)
        self.current_cost = total_manhattan_cost(self.stations, self.houses)
        self.best_ever_cost = self.current_cost
        self.algorithm_gen = None
        self._redraw_board(f"Stations reset to random positions. Initial Manhattan Distance: {self.current_cost}")

    def _create_generator(self):
        """Instantiates the generator for the active algorithm configuration."""
        algo = self.algo_var.get()

        if algo == "Hill Climbing":
            use_beam = self.beam_var.get()
            use_restart = self.restart_var.get()

            restarts = 10
            if use_restart:
                try:
                    restarts = int(self.entry_restarts.get().strip())
                    if restarts < 5 or restarts > 20:
                        raise ValueError()
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Restarts must be an integer between 5 and 20.")
                    self.entry_restarts.delete(0, tk.END)
                    self.entry_restarts.insert(0, "10")
                    restarts = 10

            beam_w = 1
            if use_beam:
                try:
                    beam_w = int(self.entry_beam.get().strip())
                    if beam_w < 2 or beam_w > 5:
                        raise ValueError()
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Beam width must be an integer between 2 and 5.")
                    self.entry_beam.delete(0, tk.END)
                    self.entry_beam.insert(0, "3")
                    beam_w = 3

            # If Random Restart is checked, run with restarts (supporting beam search if beam_w > 1)
            if use_restart:
                return random_restart_hill_climbing_generator(
                    self.stations, self.houses, restarts=restarts, beam_width=beam_w
                )
            elif use_beam:
                return local_beam_search_generator(self.stations, self.houses, beam_width=beam_w)
            else:
                return hill_climbing_generator(self.stations, self.houses)

        else:
            # Simulated Annealing
            try:
                max_t = float(self.entry_temp.get().strip())
                if max_t <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showwarning("Invalid Input", "Max temperature must be a positive number.")
                max_t = 100.0

            try:
                cooling = float(self.entry_cooling.get().strip())
                if not (0.0 < cooling < 1.0):
                    raise ValueError()
            except ValueError:
                messagebox.showwarning("Invalid Input", "Cooling rate must be a float between 0 and 1 (e.g. 0.95).")
                cooling = 0.95

            return simulated_annealing_generator(self.stations, self.houses, max_temp=max_t, cooling_rate=cooling)

    def step_algorithm(self):
        """Advances the optimization by exactly one step."""
        if self.algorithm_gen is None:
            self.algorithm_gen = self._create_generator()

        try:
            res = next(self.algorithm_gen)
            if len(res) == 5:
                stations, cost, msg, is_finished, best_cost = res
                self.best_ever_cost = best_cost
            else:
                stations, cost, msg, is_finished = res
                if self.best_ever_cost is None or cost < self.best_ever_cost:
                    self.best_ever_cost = cost

            self.stations = stations
            self.current_cost = cost
            self._redraw_board(msg)
            if is_finished:
                self.stop_run()
                self.algorithm_gen = None
            return not is_finished
        except StopIteration:
            self.stop_run()
            self.algorithm_gen = None
            self._redraw_board(f"Optimization finished! Final Manhattan Distance: {self.current_cost}")
            return False

    def toggle_run(self):
        """Toggles continuous execution between Run and Pause."""
        if self.is_running:
            self.stop_run()
        else:
            self.start_run()

    def start_run(self):
        """Starts animated continuous run."""
        self.is_running = True
        self.btn_run.config(text="Pause", bg="#DC3545")
        self._run_loop()

    def stop_run(self):
        """Pauses animated run."""
        self.is_running = False
        self.btn_run.config(text="Run", bg="#28A745")

    def _run_loop(self):
        """Recursive loop using root.after for smooth animation without freezing."""
        if not self.is_running:
            return

        has_more = self.step_algorithm()
        if has_more and self.is_running:
            self.root.after(self.animation_speed_ms, self._run_loop)
        else:
            self.stop_run()

    def _redraw_board(self, status_msg=""):
        """Draws the 20x8 grid, 9 houses, and 3 charging stations."""
        self.ax.clear()
        self.ax.set_facecolor(BACKGROUND_COLOR)

        # Set grid dimensions
        self.ax.set_xlim(0, GRID_COLS)
        self.ax.set_ylim(0, GRID_ROWS)
        self.ax.set_xticks(range(GRID_COLS + 1))
        self.ax.set_yticks(range(GRID_ROWS + 1))
        self.ax.grid(color=GRID_LINE_COLOR, linewidth=1.5)
        self.ax.set_aspect('equal')

        # Hide tick marks and numbers
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # White borders
        for spine in self.ax.spines.values():
            spine.set_edgecolor(GRID_LINE_COLOR)
            spine.set_linewidth(1.5)

        # Render 9 Houses (Bright Cyan unicode symbol)
        for hx, hy in self.houses:
            self.ax.text(
                hx + 0.5, hy + 0.5,
                HOUSE_SYMBOL,
                color=HOUSE_COLOR,
                fontsize=HOUSE_FONTSIZE,
                ha='center',
                va='center',
                fontname='Segoe UI Symbol',
                fontweight='bold'
            )

        # Render 3 Charging Stations (Bright Gold/Yellow unicode symbol)
        for sx, sy in self.stations:
            self.ax.text(
                sx + 0.5, sy + 0.5,
                STATION_SYMBOL,
                color=STATION_COLOR,
                fontsize=STATION_FONTSIZE,
                ha='center',
                va='center',
                fontname='Segoe UI Symbol',
                fontweight='bold'
            )

        # Update GUI status label with current and best-ever cost
        if hasattr(self, 'best_ever_cost') and self.best_ever_cost is not None and self.best_ever_cost != self.current_cost:
            cost_display = f"Current: {self.current_cost} (Best Ever: {self.best_ever_cost})"
        else:
            cost_display = f"{self.current_cost}"

        self.status_label.config(
            text=f"Total Manhattan Distance: {cost_display}  |  {status_msg}"
        )

        self.canvas.draw_idle()


# =====================================================================
# SECTION 5: APPLICATION ENTRY POINT
# =====================================================================
def main():
    root = tk.Tk()
    app = ChargingStationVisualizerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
