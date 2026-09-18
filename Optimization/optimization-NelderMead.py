"""
Optimization 270: Nelder-Mead Simplex Algorithm Visualizer
==========================================================
Interactive visualization of the Nelder-Mead downhill/uphill simplex optimization
algorithm on a multi-modal 2D heightmap landscape (maximizing elevation).

Problem Space:
- 2D continuous heightmap over [0, 10] x [0, 10]
- Topography features:
    * 1 Global Maxima (Elevation ~11.9 at (7.2, 7.2))
    * 2 Local Maxima  (Elevation ~8.5 at (2.8, 2.8), Elevation ~6.5 at (2.8, 7.8))
    * 2 Valleys / Pits (Central saddle valley at (5.0, 5.0), Pit at (7.5, 2.5))

Simplex Vertices:
- Named 'u', 'v', and 'w'
- At each step, vertices are dynamically ranked:
    * Best (B)        : Vertex with highest elevation
    * Good / 2nd (G)  : Vertex with intermediate elevation
    * Worst (W)       : Vertex with lowest elevation

Nelder-Mead Operations Visualized:
1. Centroid (C) calculation from the two best vertices: C = (B + G) / 2
2. Reflection (R) across the centroid: R = C + alpha * (C - W)
3. Expansion (E) when reflection reaches a new high: E = C + gamma * (R - C)
4. Outside Contraction (C_out) when R is between W and G: C_out = C + beta * (R - C)
5. Inside Contraction (C_in) when R is worse than W: C_in = C - beta * (C - W)
6. Shrink towards Best (B) when contractions fail: P_i = B + sigma * (P_i - B)

Features:
- Step button: Advance one algorithmic step at a time with full candidate inspection
- Run / Pause button: Auto-play the optimization at a legible, slow pace
- Reset button: Return simplex to selected starting corner and clear trails
- Starting Corners:
    * Bottom-Left:  u=(0, 0), v=(1, 0), w=(0, 1)
    * Bottom-Right: u=(10, 0), v=(9, 0), w=(10, 1)
    * Top-Left:     u=(0, 10), v=(1, 10), w=(0, 9)
    * Top-Right:    u=(10, 10), v=(9, 10), w=(10, 9)
    * Interactive click anywhere on the canvas to place a custom starting simplex!
- Visible Simplex: Translucent filled polygon, bold edges, labeled vertices u, v, w
- Visible Candidate Points: Reflection (R), Expansion (E), Contractions (C_out, C_in),
  with dashed projection rays and Accepted / Rejected status indicators
- Decision Explanation Panel: Plain-English breakdown explaining WHY each decision
  was made, along with exact mathematical inequalities.
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe


# =====================================================================
# SECTION 1: CONSTANTS & PARAMETERS
# =====================================================================
DOMAIN_MIN = 0.0
DOMAIN_MAX = 10.0

# Standard Nelder-Mead coefficients
ALPHA = 1.0   # Reflection
GAMMA = 2.0   # Expansion
BETA = 0.5    # Contraction
SIGMA = 0.5   # Shrink

# UI Colors (Modern Dark Theme)
BG_MAIN = "#1E1E1E"
BG_PANEL = "#252526"
BG_HEADER = "#2D2D2D"
FG_TEXT = "#E0E0E0"
FG_MUTED = "#9E9E9E"

# Simplex & Candidate Markers Color Palette
COLOR_BEST = "#FFD700"     # Bright Gold
COLOR_GOOD = "#4FC3F7"     # Light Sky Blue
COLOR_WORST = "#FF5252"    # Bright Red / Coral
COLOR_CENTROID = "#FFFFFF" # Crisp White
COLOR_REFLECT = "#E040FB"  # Bright Magenta / Violet
COLOR_EXPAND = "#00E676"   # Neon Green
COLOR_CONTRACT = "#FF9100" # Deep Amber / Orange
COLOR_SHRINK = "#FF6E40"   # Deep Coral
COLOR_ACCEPTED = "#00E676" # Green highlight
COLOR_REJECTED = "#757575" # Muted Gray


# =====================================================================
# SECTION 2: HEIGHTMAP LANDSCAPE MODEL
# =====================================================================
class HeightmapLandscape:
    """
    Continuous multimodal 2D terrain with 1 Global Maxima, 2 Local Maxima,
    and 2 valleys/pits designed for clear, educational optimization behavior.
    Includes boundary penalty to naturally reflect/contract away from domain edges.
    """

    def __init__(self):
        # Peaks: (x, y, amplitude, sigma, label)
        self.peaks = [
            (7.2, 7.2, 10.0, 1.8, "Global Maxima (H=11.9)"),
            (2.8, 2.8, 7.0, 1.6, "Local Maxima 1 (H=8.5)"),
            (2.8, 7.8, 5.8, 1.5, "Local Maxima 2 (H=6.5)"),
        ]
        # Valleys: (x, y, depth, sigma, label)
        self.valleys = [
            (5.0, 5.0, 4.0, 1.3, "Central Saddle Valley"),
            (7.5, 2.5, 3.2, 1.4, "South-East Valley"),
        ]

        # Precompute grid for smooth contour rendering
        self.res = 120
        self.x_vals = np.linspace(DOMAIN_MIN, DOMAIN_MAX, self.res)
        self.y_vals = np.linspace(DOMAIN_MIN, DOMAIN_MAX, self.res)
        self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals)
        self.Z = self._raw_evaluate(self.X, self.Y)

    def _raw_evaluate(self, x, y):
        """Calculates internal topographic function values."""
        val = 1.0 + 0.15 * (x + y) - 0.01 * (x**2 + y**2)
        for px, py, h, s, _ in self.peaks:
            val = val + h * np.exp(-((x - px)**2 + (y - py)**2) / (2.0 * s**2))
        for vx, vy, d, s, _ in self.valleys:
            val = val - d * np.exp(-((x - vx)**2 + (y - vy)**2) / (2.0 * s**2))
        return val

    def evaluate(self, x, y):
        """
        Evaluates elevation with boundary penalties so points attempting
        to move out of bounds are naturally repelled back inside.
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            raw = self._raw_evaluate(x, y)
            penalty = np.where((x < 0) | (x > 10) | (y < 0) | (y > 10), -100.0, 0.0)
            return raw + penalty

        # Scalar evaluation
        if x < DOMAIN_MIN or x > DOMAIN_MAX or y < DOMAIN_MIN or y > DOMAIN_MAX:
            dist = max(0.0, -x, x - DOMAIN_MAX) + max(0.0, -y, y - DOMAIN_MAX)
            return -100.0 - 50.0 * dist

        return float(self._raw_evaluate(x, y))


# =====================================================================
# SECTION 3: NELDER-MEAD OPTIMIZATION ENGINE
# =====================================================================
class NelderMeadOptimizer:
    """
    Engine that maintains simplex vertices 'u', 'v', 'w' and executes
    individual Nelder-Mead steps for MAXIMIZATION with full trial point telemetry.
    """

    def __init__(self, landscape: HeightmapLandscape):
        self.landscape = landscape
        self.step_count = 0
        self.points = {}  # {'u': np.array([x, y]), 'v': ..., 'w': ...}
        self.history = []  # List of past step records
        self.is_converged = False

    def reset_to_corner(self, corner_key: str):
        """Initializes simplex vertices u, v, w based on the selected corner."""
        self.step_count = 0
        self.history.clear()
        self.is_converged = False

        if corner_key == "Bottom-Left":
            # Explicitly: (0,0), (1,0), (0,1)
            self.points = {
                'u': np.array([0.0, 0.0]),
                'v': np.array([1.0, 0.0]),
                'w': np.array([0.0, 1.0]),
            }
        elif corner_key == "Bottom-Right":
            self.points = {
                'u': np.array([10.0, 0.0]),
                'v': np.array([9.0, 0.0]),
                'w': np.array([10.0, 1.0]),
            }
        elif corner_key == "Top-Left":
            self.points = {
                'u': np.array([0.0, 10.0]),
                'v': np.array([1.0, 10.0]),
                'w': np.array([0.0, 9.0]),
            }
        elif corner_key == "Top-Right":
            self.points = {
                'u': np.array([10.0, 10.0]),
                'v': np.array([9.0, 10.0]),
                'w': np.array([10.0, 9.0]),
            }
        else:
            self.points = {
                'u': np.array([0.0, 0.0]),
                'v': np.array([1.0, 0.0]),
                'w': np.array([0.0, 1.0]),
            }

    def reset_to_custom_center(self, cx: float, cy: float):
        """Initializes a 1-unit simplex centered near clicked (cx, cy)."""
        self.step_count = 0
        self.history.clear()
        self.is_converged = False

        x0 = np.clip(cx - 0.5, DOMAIN_MIN, DOMAIN_MAX - 1.0)
        y0 = np.clip(cy - 0.5, DOMAIN_MIN, DOMAIN_MAX - 1.0)

        self.points = {
            'u': np.array([x0, y0]),
            'v': np.array([x0 + 1.0, y0]),
            'w': np.array([x0, y0 + 1.0]),
        }

    def get_ranked_vertices(self):
        """
        Ranks vertices by objective function (MAXIMIZATION).
        Returns:
            best_key, good_key, worst_key
        """
        ranked = sorted(
            self.points.keys(),
            key=lambda k: self.landscape.evaluate(self.points[k][0], self.points[k][1]),
            reverse=True
        )
        return ranked[0], ranked[1], ranked[2]

    def compute_simplex_metrics(self):
        """Computes geometric area, perimeter, and elevation range."""
        u, v, w = self.points['u'], self.points['v'], self.points['w']
        area = 0.5 * abs(u[0] * (v[1] - w[1]) + v[0] * (w[1] - u[1]) + w[0] * (u[1] - v[1]))
        perim = (
            np.linalg.norm(u - v)
            + np.linalg.norm(v - w)
            + np.linalg.norm(w - u)
        )
        vals = [self.landscape.evaluate(p[0], p[1]) for p in [u, v, w]]
        spread = max(vals) - min(vals)
        return area, perim, spread

    def step(self):
        """
        Performs one full Nelder-Mead step (reflection, expansion, contraction, or shrink).
        Records all evaluated trial points, decisions, inequalities, and explanations.
        """
        if self.is_converged:
            return None

        self.step_count += 1

        # 1. Rank current vertices
        best_k, good_k, worst_k = self.get_ranked_vertices()
        B = self.points[best_k].copy()
        G = self.points[good_k].copy()
        W = self.points[worst_k].copy()

        fB = float(self.landscape.evaluate(B[0], B[1]))
        fG = float(self.landscape.evaluate(G[0], G[1]))
        fW = float(self.landscape.evaluate(W[0], W[1]))

        points_before = {k: v.copy() for k, v in self.points.items()}
        ranks_before = {best_k: 'BEST', good_k: 'GOOD', worst_k: 'WORST'}

        # 2. Centroid of all points except the worst
        C = (B + G) / 2.0
        fC = float(self.landscape.evaluate(C[0], C[1]))

        # 3. Reflection
        R = C + ALPHA * (C - W)
        fR = float(self.landscape.evaluate(R[0], R[1]))

        trial_candidates = {}
        trial_candidates['R'] = {
            'point': R,
            'val': fR,
            'accepted': False,
            'label': f"Reflection R (f={fR:.2f})"
        }

        action_name = ""
        inequality_summary = ""
        rationale_text = ""
        replaced_vertex = worst_k
        shrink_vectors = None

        # -------------------------------------------------------------
        # Decision Logic (MAXIMIZATION)
        # -------------------------------------------------------------
        if fR > fB:
            # Reflection reached a new maximum -> Test Expansion
            E = C + GAMMA * (R - C)
            fE = float(self.landscape.evaluate(E[0], E[1]))

            trial_candidates['E'] = {
                'point': E,
                'val': fE,
                'accepted': False,
                'label': f"Expansion E (f={fE:.2f})"
            }

            if fE > fR:
                # Expansion successful
                self.points[worst_k] = E
                action_name = "EXPANSION"
                trial_candidates['E']['accepted'] = True
                inequality_summary = f"f(R)={fR:.2f} > f(B)={fB:.2f}  AND  f(E)={fE:.2f} > f(R)={fR:.2f}"
                rationale_text = (
                    f"Reflection point R reached elevation {fR:.2f}, surpassing our current Best vertex "
                    f"('{best_k}' at {fB:.2f}). This indicates a strong uphill gradient! The algorithm probed "
                    f"further along this ray with Expansion point E. Because E achieved an even higher elevation "
                    f"({fE:.2f} > {fR:.2f}), Expansion was ACCEPTED. Worst vertex '{worst_k}' was replaced by E."
                )
            else:
                # Expansion did not beat reflection -> Keep Reflection
                self.points[worst_k] = R
                action_name = "REFLECTION"
                trial_candidates['R']['accepted'] = True
                inequality_summary = f"f(R)={fR:.2f} > f(B)={fB:.2f}  BUT  f(E)={fE:.2f} <= f(R)={fR:.2f}"
                rationale_text = (
                    f"Reflection point R reached elevation {fR:.2f}, higher than current Best ('{best_k}' at {fB:.2f}). "
                    f"An Expansion point E was evaluated, but its height ({fE:.2f}) did not exceed R ({fR:.2f}). "
                    f"Therefore, Reflection R was ACCEPTED and replaces Worst vertex '{worst_k}'."
                )

        elif fR >= fG:
            # Reflection is between Good and Best -> Accept Reflection
            self.points[worst_k] = R
            action_name = "REFLECTION"
            trial_candidates['R']['accepted'] = True
            inequality_summary = f"f(G)={fG:.2f} <= f(R)={fR:.2f} <= f(B)={fB:.2f}"
            rationale_text = (
                f"Reflection point R ({fR:.2f}) is higher than Second-Best ('{good_k}' at {fG:.2f}) but does not exceed "
                f"Best ('{best_k}' at {fB:.2f}). This represents steady uphill progress without overshooting. "
                f"Reflection R was ACCEPTED and replaces Worst vertex '{worst_k}'."
            )

        elif fR > fW:
            # Outside Contraction: R is better than Worst, but worse than Good
            C_out = C + BETA * (R - C)
            fC_out = float(self.landscape.evaluate(C_out[0], C_out[1]))

            trial_candidates['C_out'] = {
                'point': C_out,
                'val': fC_out,
                'accepted': False,
                'label': f"Outside Contraction C_out (f={fC_out:.2f})"
            }

            if fC_out >= fR:
                self.points[worst_k] = C_out
                action_name = "OUTSIDE CONTRACTION"
                trial_candidates['C_out']['accepted'] = True
                inequality_summary = f"f(W)={fW:.2f} < f(R)={fR:.2f} < f(G)={fG:.2f}  AND  f(C_out)={fC_out:.2f} >= f(R)={fR:.2f}"
                rationale_text = (
                    f"Reflection point R ({fR:.2f}) improved upon Worst ('{worst_k}' at {fW:.2f}), but fell short "
                    f"of Second-Best ('{good_k}' at {fG:.2f}). This suggests the summit is near the centroid. "
                    f"Outside Contraction C_out was tested halfway between Centroid and R. "
                    f"Since f(C_out)={fC_out:.2f} >= f(R), C_out was ACCEPTED, replacing '{worst_k}'."
                )
            else:
                # Contraction failed -> Shrink
                shrink_vectors = self._shrink(best_k, B)
                action_name = "SHRINK"
                replaced_vertex = f"All except Best '{best_k}'"
                inequality_summary = f"f(C_out)={fC_out:.2f} < f(R)={fR:.2f} -> Outside Contraction failed"
                rationale_text = (
                    f"Outside Contraction C_out ({fC_out:.2f}) failed to improve upon Reflection R ({fR:.2f}). "
                    f"Because neither reflection nor contraction succeeded, the algorithm executed a SHRINK: "
                    f"vertices '{good_k}' and '{worst_k}' are pulled 50% closer towards Best vertex '{best_k}'."
                )

        else:
            # Inside Contraction: R is worse than Worst
            C_in = C - BETA * (C - W)
            fC_in = float(self.landscape.evaluate(C_in[0], C_in[1]))

            trial_candidates['C_in'] = {
                'point': C_in,
                'val': fC_in,
                'accepted': False,
                'label': f"Inside Contraction C_in (f={fC_in:.2f})"
            }

            if fC_in > fW:
                self.points[worst_k] = C_in
                action_name = "INSIDE CONTRACTION"
                trial_candidates['C_in']['accepted'] = True
                inequality_summary = f"f(R)={fR:.2f} <= f(W)={fW:.2f}  AND  f(C_in)={fC_in:.2f} > f(W)={fW:.2f}"
                rationale_text = (
                    f"Reflection point R ({fR:.2f}) was worse than the Worst vertex ('{worst_k}' at {fW:.2f}), "
                    f"meaning reflection stepped into lower terrain or beyond the domain. An Inside Contraction C_in "
                    f"was evaluated between Centroid and Worst. Since f(C_in)={fC_in:.2f} > f(W), C_in was ACCEPTED "
                    f"and replaces Worst vertex '{worst_k}'."
                )
            else:
                # Inside Contraction failed -> Shrink
                shrink_vectors = self._shrink(best_k, B)
                action_name = "SHRINK"
                replaced_vertex = f"All except Best '{best_k}'"
                inequality_summary = f"f(C_in)={fC_in:.2f} <= f(W)={fW:.2f} -> Inside Contraction failed"
                rationale_text = (
                    f"Inside Contraction C_in ({fC_in:.2f}) failed to improve upon Worst vertex '{worst_k}' ({fW:.2f}). "
                    f"Because no improvement was found along the reflection ray, the algorithm executed a SHRINK: "
                    f"vertices '{good_k}' and '{worst_k}' are pulled 50% closer towards Best vertex '{best_k}'."
                )

        # Check convergence: requires small area AND small spread after initial steps
        area, perim, spread = self.compute_simplex_metrics()
        if self.step_count >= 5 and (area < 1e-4 and spread < 1e-3):
            self.is_converged = True

        step_record = {
            'step_number': self.step_count,
            'points_before': points_before,
            'ranks_before': ranks_before,
            'best_k': best_k,
            'good_k': good_k,
            'worst_k': worst_k,
            'centroid': C,
            'fC': fC,
            'candidates': trial_candidates,
            'action_name': action_name,
            'replaced_vertex': replaced_vertex,
            'inequality_summary': inequality_summary,
            'rationale_text': rationale_text,
            'shrink_vectors': shrink_vectors,
            'points_after': {k: v.copy() for k, v in self.points.items()},
            'area': area,
            'perim': perim,
            'spread': spread,
            'is_converged': self.is_converged
        }
        self.history.append(step_record)
        return step_record

    def _shrink(self, best_k, B):
        """Contracts all non-best vertices towards B by SIGMA and returns displacement vectors."""
        vectors = {}
        for k in self.points.keys():
            if k != best_k:
                old_pt = self.points[k].copy()
                new_pt = B + SIGMA * (old_pt - B)
                self.points[k] = new_pt
                vectors[k] = (old_pt, new_pt)
        return vectors


# =====================================================================
# SECTION 4: GUI & VISUALIZATION (TKINTER + EMBEDDED MATPLOTLIB)
# =====================================================================
class NelderMeadApp:
    """
    Main Visualizer Application managing Tkinter controls, Matplotlib canvas,
    simplex rendering, trial candidate visual aids, and step explanations.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Nelder-Mead Simplex Optimization Visualizer (270)")
        self.root.geometry("1420x880")
        self.root.minsize(1200, 750)
        self.root.configure(bg=BG_MAIN)

        # State initialization
        self.landscape = HeightmapLandscape()
        self.optimizer = NelderMeadOptimizer(self.landscape)
        self.is_running = False
        self.auto_run_job = None
        self.step_delay_ms = 700  # Default slow, legible pace
        self.show_trail_var = tk.BooleanVar(value=True)

        # Default to Bottom-Left corner
        self.current_corner = "Bottom-Left"
        self.optimizer.reset_to_corner(self.current_corner)

        # Build UI layout
        self._build_gui()

        # Initial render
        self.last_step_record = None
        self._update_plot()
        self._update_explanation_panel(None)

    def _build_gui(self):
        """Constructs layout: top controls, center Matplotlib canvas, right explanation panel."""
        # Configure ttk styling so colored buttons display properly across OSes (especially macOS Aqua)
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.style.configure("NMStep.TButton", background="#0288D1", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"), borderwidth=1, padding=(12, 4))
        self.style.map("NMStep.TButton", background=[("active", "#03A9F4")], foreground=[("active", "#FFFFFF")])

        self.style.configure("NMRun.TButton", background="#2E7D32", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"), borderwidth=1, padding=(14, 4))
        self.style.map("NMRun.TButton", background=[("active", "#388E3C")], foreground=[("active", "#FFFFFF")])

        self.style.configure("NMPause.TButton", background="#E65100", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"), borderwidth=1, padding=(14, 4))
        self.style.map("NMPause.TButton", background=[("active", "#F57C00")], foreground=[("active", "#FFFFFF")])

        self.style.configure("NMReset.TButton", background="#4A4A4A", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"), borderwidth=1, padding=(12, 4))
        self.style.map("NMReset.TButton", background=[("active", "#606060")], foreground=[("active", "#FFFFFF")])

        # 1. Top Control Bar
        top_bar = tk.Frame(self.root, bg=BG_HEADER, padx=12, pady=8, relief=tk.RAISED, bd=1)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # Starting Corner Selection
        tk.Label(
            top_bar,
            text="Starting Simplex:",
            fg="#FFFFFF",
            bg=BG_HEADER,
            font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT, padx=(5, 5))

        self.corner_combo = ttk.Combobox(
            top_bar,
            values=[
                "Bottom-Left: (0,0), (1,0), (0,1)",
                "Bottom-Right: (10,0), (9,0), (10,1)",
                "Top-Left: (0,10), (1,10), (0,9)",
                "Top-Right: (10,10), (9,10), (10,9)"
            ],
            state="readonly",
            width=32,
            font=("Segoe UI", 9)
        )
        self.corner_combo.current(0)
        self.corner_combo.pack(side=tk.LEFT, padx=5)
        self.corner_combo.bind("<<ComboboxSelected>>", self._on_corner_change)

        # Separator
        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Step Button
        self.btn_step = ttk.Button(
            top_bar,
            text="▶ Step (1 Step)",
            command=self.step_forward,
            style="NMStep.TButton"
        )
        self.btn_step.pack(side=tk.LEFT, padx=4)

        # Run / Pause Button
        self.btn_run = ttk.Button(
            top_bar,
            text="▶ Run Full Simulation",
            command=self.toggle_run,
            style="NMRun.TButton"
        )
        self.btn_run.pack(side=tk.LEFT, padx=4)

        # Reset Button
        self.btn_reset = ttk.Button(
            top_bar,
            text="⟲ Reset",
            command=self.reset_simulation,
            style="NMReset.TButton"
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        # Separator
        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Speed Control
        tk.Label(
            top_bar,
            text="Pace:",
            fg="#FFFFFF",
            bg=BG_HEADER,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(5, 2))

        self.speed_combo = ttk.Combobox(
            top_bar,
            values=["Slow (1.0s)", "Legible (0.7s)", "Medium (0.4s)", "Fast (0.15s)"],
            state="readonly",
            width=14,
            font=("Segoe UI", 9)
        )
        self.speed_combo.current(1)
        self.speed_combo.pack(side=tk.LEFT, padx=4)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # History Trail Checkbox
        self.chk_trail = tk.Checkbutton(
            top_bar,
            text="Show History Trail",
            variable=self.show_trail_var,
            command=self._on_trail_toggle,
            bg=BG_HEADER,
            fg="#FFFFFF",
            selectcolor="#333333",
            activebackground=BG_HEADER,
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9)
        )
        self.chk_trail.pack(side=tk.LEFT, padx=12)

        # Interactive Hint label
        tk.Label(
            top_bar,
            text="Tip: Click anywhere on the map to place a custom simplex!",
            fg="#FFD54F",
            bg=BG_HEADER,
            font=("Segoe UI", 8, "italic")
        ).pack(side=tk.RIGHT, padx=10)

        # 2. Main Content Split (Matplotlib on Left, Explanation on Right)
        content_frame = tk.Frame(self.root, bg=BG_MAIN)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Left: Matplotlib Canvas Frame
        plot_frame = tk.Frame(content_frame, bg=BG_PANEL, relief=tk.GROOVE, bd=1)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.fig = Figure(figsize=(8.2, 7.2), dpi=100, facecolor=BG_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect click event for custom starting position
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # Right: Detail & Explanation Panel (Width ~470px)
        self.panel_frame = tk.Frame(content_frame, bg=BG_PANEL, width=470, relief=tk.GROOVE, bd=1)
        self.panel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.panel_frame.pack_propagate(False)

        self._build_panel_widgets()

    def _build_panel_widgets(self):
        """Constructs right-hand explanation, status cards, and decision log."""
        # Top title badge
        header = tk.Frame(self.panel_frame, bg="#333333", padx=10, pady=6)
        header.pack(fill=tk.X)
        self.lbl_step_title = tk.Label(
            header,
            text="Step 0: Initial Simplex Ready",
            fg="#FFFFFF",
            bg="#333333",
            font=("Segoe UI", 12, "bold")
        )
        self.lbl_step_title.pack(side=tk.LEFT)

        self.lbl_status_badge = tk.Label(
            header,
            text="READY",
            fg="#00E676",
            bg="#1B5E20",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=1
        )
        self.lbl_status_badge.pack(side=tk.RIGHT)

        # Scrollable container for explanations and history
        container = tk.Frame(self.panel_frame, bg=BG_PANEL)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # --- Section A: Vertices Card ---
        sec_a = tk.LabelFrame(
            container,
            text=" Simplex Vertices (u, v, w) ",
            fg="#64B5F6",
            bg=BG_PANEL,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=5
        )
        sec_a.pack(fill=tk.X, pady=(0, 6))

        self.lbl_u = tk.Label(sec_a, text="u: (0.00, 0.00) | H=0.00", fg=FG_TEXT, bg=BG_PANEL, font=("Consolas", 9), anchor="w")
        self.lbl_u.pack(fill=tk.X)
        self.lbl_v = tk.Label(sec_a, text="v: (1.00, 0.00) | H=0.00", fg=FG_TEXT, bg=BG_PANEL, font=("Consolas", 9), anchor="w")
        self.lbl_v.pack(fill=tk.X)
        self.lbl_w = tk.Label(sec_a, text="w: (0.00, 1.00) | H=0.00", fg=FG_TEXT, bg=BG_PANEL, font=("Consolas", 9), anchor="w")
        self.lbl_w.pack(fill=tk.X)

        self.lbl_metrics = tk.Label(
            sec_a,
            text="Spread Δf: 0.00 | Area: 0.50 | Perim: 3.41",
            fg=FG_MUTED,
            bg=BG_PANEL,
            font=("Segoe UI", 8, "italic"),
            anchor="w"
        )
        self.lbl_metrics.pack(fill=tk.X, pady=(3, 0))

        # --- Section B: Current Step Decision & Inequality ---
        sec_b = tk.LabelFrame(
            container,
            text=" Decision & Inequality Tested ",
            fg="#FFB74D",
            bg=BG_PANEL,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=5
        )
        sec_b.pack(fill=tk.X, pady=(0, 6))

        self.lbl_action_badge = tk.Label(
            sec_b,
            text="Action: Awaiting First Step",
            fg="#FFD54F",
            bg=BG_PANEL,
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        )
        self.lbl_action_badge.pack(fill=tk.X)

        self.lbl_centroid = tk.Label(
            sec_b,
            text="Centroid C = (—, —)",
            fg=FG_MUTED,
            bg=BG_PANEL,
            font=("Consolas", 8),
            anchor="w"
        )
        self.lbl_centroid.pack(fill=tk.X)

        self.lbl_inequality = tk.Label(
            sec_b,
            text="Condition: Press 'Step' or 'Run' to begin.",
            fg="#81C784",
            bg=BG_PANEL,
            font=("Consolas", 8),
            anchor="w",
            wraplength=430,
            justify="left"
        )
        self.lbl_inequality.pack(fill=tk.X, pady=2)

        # --- Section C: Plain English Rationale ("Why this decision was made") ---
        sec_c = tk.LabelFrame(
            container,
            text=" Why This Decision Was Made ",
            fg="#81C784",
            bg=BG_PANEL,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=5
        )
        sec_c.pack(fill=tk.X, pady=(0, 6))

        self.txt_rationale = tk.Text(
            sec_c,
            height=5,
            bg="#1E1E1E",
            fg="#FFFFFF",
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=6,
            pady=4
        )
        self.txt_rationale.pack(fill=tk.BOTH, expand=True)
        self.txt_rationale.insert(
            tk.END,
            "The simplex is ready at the starting corner. Vertices u, v, w form a triangle on the "
            "heightmap. In each step, the algorithm identifies Best, Good, and Worst vertices, computes "
            "the Centroid of the two best, and explores uphill by reflecting across the centroid."
        )
        self.txt_rationale.config(state=tk.DISABLED)

        # --- Section D: Step History Log ---
        sec_d = tk.LabelFrame(
            container,
            text=" Step-by-Step History Log ",
            fg="#BA68C8",
            bg=BG_PANEL,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=5
        )
        sec_d.pack(fill=tk.BOTH, expand=True)

        history_scroll = ttk.Scrollbar(sec_d)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_history = tk.Text(
            sec_d,
            bg="#1E1E1E",
            fg="#CE93D8",
            font=("Consolas", 8),
            wrap=tk.NONE,
            relief=tk.FLAT,
            yscrollcommand=history_scroll.set
        )
        self.txt_history.pack(fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.txt_history.yview)

    # -------------------------------------------------------------
    # Plot Rendering
    # -------------------------------------------------------------
    def _update_plot(self):
        """Renders the heightmap contours, peaks, simplex, centroid, candidate points, and trail."""
        self.ax.clear()

        # 1. Filled contour map
        cf = self.ax.contourf(
            self.landscape.X,
            self.landscape.Y,
            self.landscape.Z,
            levels=25,
            cmap="viridis",
            alpha=0.88
        )

        # 2. Subtle contour lines with elevation labels
        cs = self.ax.contour(
            self.landscape.X,
            self.landscape.Y,
            self.landscape.Z,
            levels=12,
            colors="white",
            alpha=0.25,
            linewidths=0.6
        )
        self.ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f", colors="#DDDDDD")

        # 3. Peak and Valley markers
        for px, py, h, _, label in self.landscape.peaks:
            is_global = "Global" in label
            marker = "*" if is_global else "^"
            color = "#FFD700" if is_global else "#FFA726"
            size = 180 if is_global else 100
            self.ax.scatter(px, py, color=color, s=size, marker=marker, zorder=5, edgecolors="black", linewidths=1.2)
            self.ax.text(
                px + 0.18, py + 0.18, label,
                color="#FFFFFF", fontsize=8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
            )

        for vx, vy, _, _, label in self.landscape.valleys:
            self.ax.scatter(vx, vy, color="#29B6F6", s=80, marker="v", zorder=5, edgecolors="black")
            self.ax.text(
                vx + 0.18, vy - 0.28, f"[Valley] {label}",
                color="#B3E5FC", fontsize=7.5, zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            )

        # 4. History Trail (if enabled)
        if self.show_trail_var.get() and len(self.optimizer.history) > 1:
            centroids = [rec['centroid'] for rec in self.optimizer.history]
            cx_vals = [c[0] for c in centroids]
            cy_vals = [c[1] for c in centroids]
            self.ax.plot(
                cx_vals, cy_vals,
                linestyle=":", color="#FFFFFF", alpha=0.55, linewidth=1.5, zorder=3,
                label="Centroid Trajectory"
            )

        # 5. Candidate Points and Construction Lines from Last Step
        if self.last_step_record is not None:
            rec = self.last_step_record
            C = rec['centroid']
            points_before = rec['points_before']
            worst_pt = points_before[rec['worst_k']]

            # Draw dashed construction line from Worst through Centroid to furthest candidate
            candidates = rec['candidates']
            furthest_pt = C
            if 'E' in candidates:
                furthest_pt = candidates['E']['point']
            elif 'R' in candidates:
                furthest_pt = candidates['R']['point']

            self.ax.plot(
                [worst_pt[0], furthest_pt[0]],
                [worst_pt[1], furthest_pt[1]],
                linestyle="--", color="#E0E0E0", alpha=0.8, linewidth=1.4, zorder=4
            )

            # Draw Centroid marker C
            self.ax.scatter(C[0], C[1], color=COLOR_CENTROID, s=85, marker="D", zorder=7, edgecolors="black", linewidths=1.2)

            # Draw Candidate trial points
            for c_key, c_data in candidates.items():
                pt = c_data['point']
                val = c_data['val']
                is_acc = c_data['accepted']

                if c_key == 'R':
                    c_color = COLOR_REFLECT
                elif c_key == 'E':
                    c_color = COLOR_EXPAND
                else:
                    c_color = COLOR_CONTRACT

                if is_acc:
                    self.ax.scatter(pt[0], pt[1], color=c_color, s=150, zorder=8, edgecolors="#00E676", linewidths=2.5)
                else:
                    self.ax.scatter(pt[0], pt[1], color=c_color, s=80, marker="x", zorder=7, linewidths=2.0)

            # If shrink occurred, draw displacement arrows
            if rec.get('shrink_vectors'):
                for v_name, (v_start, v_end) in rec['shrink_vectors'].items():
                    self.ax.annotate(
                        "",
                        xy=(v_end[0], v_end[1]),
                        xytext=(v_start[0], v_start[1]),
                        arrowprops=dict(arrowstyle="->", color=COLOR_SHRINK, lw=1.8, ls=":")
                    )

        # 6. Current Simplex Triangle & Vertices u, v, w
        pts = self.optimizer.points
        u, v, w = pts['u'], pts['v'], pts['w']
        triangle_coords = np.array([u, v, w])

        # Filled translucent polygon
        simplex_patch = Polygon(
            triangle_coords,
            closed=True,
            facecolor="#00E5FF",
            edgecolor="#FFFFFF",
            alpha=0.30,
            linewidth=2.2,
            zorder=6
        )
        self.ax.add_patch(simplex_patch)

        # Rank vertices to color code Best, Good, Worst
        ranked = sorted(pts.keys(), key=lambda k: self.landscape.evaluate(pts[k][0], pts[k][1]), reverse=True)
        ranks = {ranked[0]: ("BEST", COLOR_BEST), ranked[1]: ("GOOD", COLOR_GOOD), ranked[2]: ("WORST", COLOR_WORST)}

        # Draw vertex points
        for k in ['u', 'v', 'w']:
            coord = pts[k]
            _, r_color = ranks[k]
            self.ax.scatter(
                coord[0], coord[1],
                color=r_color,
                s=130,
                zorder=10,
                edgecolors="#FFFFFF",
                linewidths=1.5
            )

        # Set axes properties before calculating display transformations
        self.ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        self.ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        self.ax.set_aspect("equal", adjustable="box")

        # 7. Collision-Free Radial Label Placement
        # Calculate geometric center of simplex to radiate labels outwards
        M = (pts['u'] + pts['v'] + pts['w']) / 3.0
        labels_to_place = []

        # (a) Simplex Vertices u, v, w
        for k in ['u', 'v', 'w']:
            coord = pts[k]
            h = float(self.landscape.evaluate(coord[0], coord[1]))
            r_name, r_color = ranks[k]
            vec = coord - M
            dist = np.linalg.norm(vec)
            uvec = vec / dist if dist > 1e-4 else np.array([0.0, 1.0])
            labels_to_place.append({
                'xy': (float(coord[0]), float(coord[1])),
                'offset': [float(uvec[0] * 62), float(uvec[1] * 62)],
                'text': f"{k.upper()} [{r_name}]\n({coord[0]:.2f}, {coord[1]:.2f})\nH={h:.2f}",
                'color': r_color,
                'width': 98,
                'height': 44,
                'lw': 1.2
            })

        # (b) Centroid C and Candidate Points
        if self.last_step_record is not None:
            rec = self.last_step_record
            C = rec['centroid']
            vec_c = C - M
            dist_c = np.linalg.norm(vec_c)
            uvec_c = vec_c / dist_c if dist_c > 1e-4 else np.array([-1.0, 0.0])
            labels_to_place.append({
                'xy': (float(C[0]), float(C[1])),
                'offset': [float(uvec_c[0] * 58), float(uvec_c[1] * 58)],
                'text': f"C (Centroid)\n({C[0]:.2f}, {C[1]:.2f})",
                'color': '#FFFFFF',
                'width': 86,
                'height': 34,
                'lw': 1.0
            })

            worst_pt = rec['points_before'][rec['worst_k']]
            ray = C - worst_pt
            ray_len = np.linalg.norm(ray)
            ray_u = ray / ray_len if ray_len > 1e-4 else np.array([1.0, 0.0])
            normal = np.array([-ray_u[1], ray_u[0]])

            for c_key, c_data in rec['candidates'].items():
                pt = c_data['point']
                val = c_data['val']
                is_acc = c_data['accepted']
                side = 1.0 if is_acc else -1.0
                tag = f"{c_key}: f={val:.2f}"
                status = "[ACCEPTED]" if is_acc else "[REJECTED]"
                col = '#00E676' if is_acc else '#FF8A80'
                labels_to_place.append({
                    'xy': (float(pt[0]), float(pt[1])),
                    'offset': [float(normal[0] * 65 * side), float(normal[1] * 65 * side)],
                    'text': f"{tag}\n{status}",
                    'color': col,
                    'width': 88,
                    'height': 34,
                    'lw': 1.2
                })

        # Multi-pass collision avoidance relaxation in display coordinates
        for it in range(10):
            for i in range(len(labels_to_place)):
                for j in range(i + 1, len(labels_to_place)):
                    l1 = labels_to_place[i]
                    l2 = labels_to_place[j]
                    p1 = self.ax.transData.transform(l1['xy']) + np.array(l1['offset'])
                    p2 = self.ax.transData.transform(l2['xy']) + np.array(l2['offset'])
                    diff = p2 - p1
                    min_dx = (l1['width'] + l2['width']) * 0.52
                    min_dy = (l1['height'] + l2['height']) * 0.55
                    if abs(diff[0]) < min_dx and abs(diff[1]) < min_dy:
                        overlap_x = min_dx - abs(diff[0])
                        overlap_y = min_dy - abs(diff[1])
                        sign_x = 1.0 if diff[0] >= 0 else -1.0
                        sign_y = 1.0 if diff[1] >= 0 else -1.0
                        if overlap_y < overlap_x * 0.7:
                            l1['offset'][1] -= overlap_y * 0.55 * sign_y
                            l2['offset'][1] += overlap_y * 0.55 * sign_y
                        else:
                            l1['offset'][0] -= overlap_x * 0.55 * sign_x
                            l2['offset'][0] += overlap_x * 0.55 * sign_x

        # Render each collision-resolved annotation badge with directional pointer arrow
        for l in labels_to_place:
            self.ax.annotate(
                l['text'],
                xy=l['xy'],
                xytext=(l['offset'][0], l['offset'][1]),
                textcoords="offset points",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#181818", edgecolor=l['color'], alpha=0.92, linewidth=l['lw']),
                arrowprops=dict(arrowstyle="->", color=l['color'], lw=l['lw'], shrinkA=3, shrinkB=6),
                color=l['color'], fontsize=7.5, fontweight="bold", zorder=12
            )

        # Canvas properties
        self.ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        self.ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(
            f"Nelder-Mead 2D Simplex: Step {self.optimizer.step_count} (Landscape Elevation)",
            color="#FFFFFF", fontsize=11, fontweight="bold", pad=8
        )
        self.ax.tick_params(colors="#CCCCCC", labelsize=8)
        self.ax.grid(True, linestyle=":", color="#555555", alpha=0.4)

        for spine in self.ax.spines.values():
            spine.set_color("#555555")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------
    # Side Panel & Explanation Updates
    # -------------------------------------------------------------
    def _update_explanation_panel(self, step_record):
        """Updates the text labels, rationale, and history log on the right side."""
        pts = self.optimizer.points
        ranked = sorted(pts.keys(), key=lambda k: self.landscape.evaluate(pts[k][0], pts[k][1]), reverse=True)
        ranks = {ranked[0]: "BEST", ranked[1]: "GOOD", ranked[2]: "WORST"}

        # Update Vertices Card
        for k, lbl in [('u', self.lbl_u), ('v', self.lbl_v), ('w', self.lbl_w)]:
            p = pts[k]
            val = float(self.landscape.evaluate(p[0], p[1]))
            r_str = ranks[k]
            lbl.config(
                text=f"{k.upper()}: ({p[0]:5.2f}, {p[1]:5.2f}) | H={val:5.2f} [{r_str}]",
                fg=COLOR_BEST if r_str == "BEST" else (COLOR_GOOD if r_str == "GOOD" else COLOR_WORST)
            )

        area, perim, spread = self.optimizer.compute_simplex_metrics()
        self.lbl_metrics.config(
            text=f"Spread Δf: {spread:.4f} | Area: {area:.4f} | Perimeter: {perim:.3f}"
        )

        if step_record is None:
            self.lbl_step_title.config(text="Step 0: Initial Simplex Ready")
            self.lbl_status_badge.config(text="READY", bg="#1B5E20", fg="#00E676")
            self.lbl_action_badge.config(text="Action: Awaiting First Step", fg="#FFD54F")
            self.lbl_centroid.config(text="Centroid C = (—, —)")
            self.lbl_inequality.config(text="Condition: Press 'Step' or 'Run' to begin.")
            return

        # Update Step Record Info
        step_num = step_record['step_number']
        action = step_record['action_name']
        C = step_record['centroid']
        fC = step_record['fC']

        self.lbl_step_title.config(text=f"Step {step_num}: {action}")
        if step_record['is_converged']:
            self.lbl_status_badge.config(text="CONVERGED", bg="#E65100", fg="#FFD54F")
        else:
            self.lbl_status_badge.config(text="IN PROGRESS", bg="#0D47A1", fg="#80D8FF")

        self.lbl_action_badge.config(text=f"Action: {action} (Accepted)", fg="#00E676")
        self.lbl_centroid.config(text=f"Centroid C: ({C[0]:.2f}, {C[1]:.2f}) | f(C)={fC:.2f}")
        self.lbl_inequality.config(text=f"Inequality: {step_record['inequality_summary']}")

        # Rationale Text
        self.txt_rationale.config(state=tk.NORMAL)
        self.txt_rationale.delete("1.0", tk.END)
        self.txt_rationale.insert(tk.END, step_record['rationale_text'])
        self.txt_rationale.config(state=tk.DISABLED)

        # Append to History Log
        self.txt_history.insert(
            tk.END,
            f"Step {step_num:02d}: {action:<18} -> Replaced {step_record['replaced_vertex']} "
            f"| Best={step_record['best_k'].upper()} ({self.landscape.evaluate(pts[step_record['best_k']][0], pts[step_record['best_k']][1]):.2f})\n"
        )
        self.txt_history.see(tk.END)

    # -------------------------------------------------------------
    # Control Callbacks
    # -------------------------------------------------------------
    def step_forward(self):
        """Advances one algorithmic step, renders updates, and checks convergence."""
        if self.optimizer.is_converged:
            if self.is_running:
                self.toggle_run()
            messagebox.showinfo("Convergence", "The Nelder-Mead simplex has converged onto a peak!")
            return False

        record = self.optimizer.step()
        if record is not None:
            self.last_step_record = record
            self._update_plot()
            self._update_explanation_panel(record)

            if record['is_converged']:
                if self.is_running:
                    self.toggle_run()
                best_k = record['best_k']
                best_pt = self.optimizer.points[best_k]
                best_val = self.landscape.evaluate(best_pt[0], best_pt[1])
                messagebox.showinfo(
                    "Convergence Reached",
                    f"Simplex converged at Step {record['step_number']}!\n"
                    f"Peak Vertex: {best_k.upper()} at ({best_pt[0]:.3f}, {best_pt[1]:.3f})\n"
                    f"Elevation: {best_val:.3f}"
                )
                return False
            return True
        return False

    def toggle_run(self):
        """Starts or pauses the automated slow, legible execution."""
        if self.is_running:
            # Pause
            self.is_running = False
            if self.auto_run_job is not None:
                self.root.after_cancel(self.auto_run_job)
                self.auto_run_job = None
            self.btn_run.config(text="▶ Resume Simulation", style="NMRun.TButton")
        else:
            # Start
            if self.optimizer.is_converged:
                self.reset_simulation()
            self.is_running = True
            self.btn_run.config(text="⏸ Pause", style="NMPause.TButton")
            self._auto_step()

    def _auto_step(self):
        """Timer callback for auto-run mode."""
        if not self.is_running:
            return

        has_next = self.step_forward()
        if has_next and self.is_running:
            self.auto_run_job = self.root.after(self.step_delay_ms, self._auto_step)
        else:
            self.is_running = False
            self.btn_run.config(text="▶ Run Full Simulation", style="NMRun.TButton")

    def reset_simulation(self):
        """Restores simplex to currently selected corner and clears history."""
        if self.is_running:
            self.toggle_run()

        corner_key = self.corner_combo.get().split(":")[0].strip()
        self.optimizer.reset_to_corner(corner_key)
        self.last_step_record = None

        # Reset history widget
        self.txt_history.delete("1.0", tk.END)

        # Reset rationale widget
        self.txt_rationale.config(state=tk.NORMAL)
        self.txt_rationale.delete("1.0", tk.END)
        self.txt_rationale.insert(
            tk.END,
            f"Reset to {corner_key} corner.\n"
            f"Points: u={tuple(np.round(self.optimizer.points['u'], 2))}, "
            f"v={tuple(np.round(self.optimizer.points['v'], 2))}, "
            f"w={tuple(np.round(self.optimizer.points['w'], 2))}.\n"
            f"Press 'Step' to inspect the first reflection or 'Run' to play automatically."
        )
        self.txt_rationale.config(state=tk.DISABLED)

        self._update_plot()
        self._update_explanation_panel(None)

    def _on_corner_change(self, event=None):
        """Triggered when user selects a different corner from dropdown."""
        self.reset_simulation()

    def _on_speed_change(self, event=None):
        """Adjusts the animation speed delay."""
        selection = self.speed_combo.get()
        if "1.0s" in selection:
            self.step_delay_ms = 1000
        elif "0.7s" in selection:
            self.step_delay_ms = 700
        elif "0.4s" in selection:
            self.step_delay_ms = 400
        elif "0.15s" in selection:
            self.step_delay_ms = 150

    def _on_trail_toggle(self):
        """Redraws the plot with or without the centroid trajectory trail."""
        self._update_plot()

    def _on_canvas_click(self, event):
        """Allows user to click anywhere on the heightmap to place a custom simplex."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        cx = float(event.xdata)
        cy = float(event.ydata)

        if self.is_running:
            self.toggle_run()

        self.optimizer.reset_to_custom_center(cx, cy)
        self.last_step_record = None
        self.txt_history.delete("1.0", tk.END)

        self.txt_rationale.config(state=tk.NORMAL)
        self.txt_rationale.delete("1.0", tk.END)
        self.txt_rationale.insert(
            tk.END,
            f"Custom simplex placed near clicked position ({cx:.2f}, {cy:.2f}).\n"
            f"Points: u={tuple(np.round(self.optimizer.points['u'], 2))}, "
            f"v={tuple(np.round(self.optimizer.points['v'], 2))}, "
            f"w={tuple(np.round(self.optimizer.points['w'], 2))}.\n"
            f"Press 'Step' or 'Run' to optimize from here."
        )
        self.txt_rationale.config(state=tk.DISABLED)

        self._update_plot()
        self._update_explanation_panel(None)


# =====================================================================
# SECTION 5: MAIN EXECUTION ENTRY POINT
# =====================================================================
def main():
    """Initializes Tkinter window and launches the application."""
    root = tk.Tk()
    app = NelderMeadApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
