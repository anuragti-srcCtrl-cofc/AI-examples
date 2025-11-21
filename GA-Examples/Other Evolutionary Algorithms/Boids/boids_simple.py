import pygame
import random
import math
import dearpygui.dearpygui as dpg
import time

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
WIDTH, HEIGHT = 1200, 700
NUM_BOIDS = 40
MAX_SPEED = 4
MAX_FORCE = 0.05
PREDATOR_RADIUS = 150  # radius around mouse for fleeing

# GUI state & rule flags (initial values)
sep_factor = 1.5
coh_factor = 1.0
ali_factor = 1.0

sep_enabled = False
coh_enabled = False
ali_enabled = False

mouse_enabled = False  # toggle for mouse as predator/prey (initially OFF)
goals_enabled = False  # toggle for random goals (initially OFF)

# GUI control tags (so we can programmatically change them)
TAG_SLIDER_SEP = "s_sep"
TAG_SLIDER_COH = "s_coh"
TAG_SLIDER_ALI = "s_ali"
TAG_CHECK_SEP = "chk_sep"
TAG_CHECK_COH = "chk_coh"
TAG_CHECK_ALI = "chk_ali"
TAG_CHECK_MOUSE = "chk_mouse"

# ---------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------
def limit(vec, max_value):
    vec = list(vec)
    mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    if mag > max_value:
        return [vec[0] / mag * max_value, vec[1] / mag * max_value]
    return vec

def dist(a, b):
    return math.dist(a, b)

# ---------------------------------------------------------------
# BOID CLASS
# ---------------------------------------------------------------
class Boid:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = [random.uniform(0, WIDTH), random.uniform(0, HEIGHT)]
        angle = random.uniform(0, math.pi * 2)
        self.vel = [math.cos(angle), math.sin(angle)]
        self.acc = [0, 0]

    def update(self):
        # ensure lists for in-place ops
        self.vel = list(self.vel)
        self.acc = list(self.acc)

        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]
        self.vel = limit(self.vel, MAX_SPEED)

        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.acc = [0, 0]

        # Wrap edges
        if self.pos[0] < 0: self.pos[0] = WIDTH
        if self.pos[0] > WIDTH: self.pos[0] = 0
        if self.pos[1] < 0: self.pos[1] = HEIGHT
        if self.pos[1] > HEIGHT: self.pos[1] = 0

    def apply_force(self, force):
        force = list(force)
        self.acc[0] += force[0]
        self.acc[1] += force[1]

    def separation(self, boids):
        desired_sep = 40
        steer = [0, 0]
        count = 0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < desired_sep:
                diff = [(self.pos[0] - b.pos[0]) / d,
                        (self.pos[1] - b.pos[1]) / d]
                steer[0] += diff[0]
                steer[1] += diff[1]
                count += 1
        if count > 0:
            steer[0] /= count
            steer[1] /= count
        return limit(steer, MAX_FORCE)

    def alignment(self, boids):
        neighbordist = 60
        sum_vel = [0, 0]
        count = 0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < neighbordist:
                sum_vel[0] += b.vel[0]
                sum_vel[1] += b.vel[1]
                count += 1
        if count > 0:
            sum_vel[0] /= count
            sum_vel[1] /= count
            desired = limit(sum_vel, MAX_SPEED)
            steer = [desired[0] - self.vel[0], desired[1] - self.vel[1]]
            return limit(steer, MAX_FORCE)
        return [0, 0]

    def cohesion(self, boids):
        neighbordist = 60
        center = [0, 0]
        count = 0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < neighbordist:
                center[0] += b.pos[0]
                center[1] += b.pos[1]
                count += 1
        if count > 0:
            center[0] /= count
            center[1] /= count
            desired = [center[0] - self.pos[0], center[1] - self.pos[1]]
            desired = limit(desired, MAX_SPEED)
            steer = [desired[0] - self.vel[0], desired[1] - self.vel[1]]
            return limit(steer, MAX_FORCE)
        return [0, 0]

    def seek(self, target, flee=False, radius=None):
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d == 0:
            return [0,0]
        if flee and radius is not None and d > radius:
            return [0,0]
        desired = [(dx/d) * MAX_SPEED, (dy/d) * MAX_SPEED]
        if flee:
            desired[0] *= -1
            desired[1] *= -1
        steer = [desired[0] - self.vel[0], desired[1] - self.vel[1]]
        return limit(steer, MAX_FORCE*2 if flee else MAX_FORCE)

    def run(self, boids, target=None, flee=False, flee_radius=None,
            apply_sep=False, apply_ali=False, apply_coh=False,
            sep_strength=1.0, ali_strength=1.0, coh_strength=1.0):
        # compute rule vectors
        if apply_sep:
            sep = self.separation(boids)
            self.apply_force([sep[0] * sep_strength, sep[1] * sep_strength])
        if apply_ali:
            ali = self.alignment(boids)
            self.apply_force([ali[0] * ali_strength, ali[1] * ali_strength])
        if apply_coh:
            coh = self.cohesion(boids)
            self.apply_force([coh[0] * coh_strength, coh[1] * coh_strength])

        if target is not None:
            self.apply_force(self.seek(target, flee, radius=flee_radius))

        self.update()

    def draw(self, screen):
        angle = math.atan2(self.vel[1], self.vel[0])
        size = 8
        points = [
            (self.pos[0] + math.cos(angle)*size, self.pos[1] + math.sin(angle)*size),
            (self.pos[0] + math.cos(angle+2.5)*size, self.pos[1] + math.sin(angle+2.5)*size),
            (self.pos[0] + math.cos(angle-2.5)*size, self.pos[1] + math.sin(angle-2.5)*size)
        ]
        pygame.draw.polygon(screen, (255,255,255), points)

# ---------------------------------------------------------------
# CALLBACKS (GUI)
# ---------------------------------------------------------------
def slider_sep_cb(sender, app_data):
    global sep_factor
    sep_factor = app_data

def slider_coh_cb(sender, app_data):
    global coh_factor
    coh_factor = app_data

def slider_ali_cb(sender, app_data):
    global ali_factor
    ali_factor = app_data

def check_sep_cb(sender, app_data):
    global sep_enabled
    sep_enabled = app_data

def check_coh_cb(sender, app_data):
    global coh_enabled
    coh_enabled = app_data

def check_ali_cb(sender, app_data):
    global ali_enabled
    ali_enabled = app_data

def mouse_toggle_callback(sender, app_data):
    global mouse_enabled
    mouse_enabled = app_data

def start_goals_callback(sender, app_data):
    global goals_enabled, next_goal_time, goal_point
    goals_enabled = True
    goal_point = [random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)]
    next_goal_time = time.time() + random.uniform(5,10)

def reset_boids_callback(sender, app_data):
    global goals_enabled, mouse_enabled, sep_enabled, coh_enabled, ali_enabled, target
    # disable systems
    goals_enabled = False
    mouse_enabled = False
    sep_enabled = False
    coh_enabled = False
    ali_enabled = False

    # reset GUI checkboxes to OFF
    try:
        dpg.set_value(TAG_CHECK_SEP, False)
        dpg.set_value(TAG_CHECK_COH, False)
        dpg.set_value(TAG_CHECK_ALI, False)
        dpg.set_value(TAG_CHECK_MOUSE, False)
    except Exception:
        # if GUI not ready or tags changed, ignore
        pass

    # reset boids
    for b in boids:
        b.reset()

    # clear target so nothing is applied until user clicks Start Random Goals or enables mouse
    target = None

# ---------------------------------------------------------------
# DEAR PYGUI UI SETUP
# ---------------------------------------------------------------
dpg.create_context()
dpg.create_viewport(title="Boids Controls", width=360, height=380)

with dpg.window(label="Controls", width=340, height=360):
    dpg.add_text("Enable rules (all OFF at start):")
    dpg.add_checkbox(label="Enable Separation", tag=TAG_CHECK_SEP, default_value=False, callback=check_sep_cb)
    dpg.add_checkbox(label="Enable Alignment", tag=TAG_CHECK_ALI, default_value=False, callback=check_ali_cb)
    dpg.add_checkbox(label="Enable Cohesion", tag=TAG_CHECK_COH, default_value=False, callback=check_coh_cb)

    dpg.add_spacing(count=1)
    dpg.add_text("Rule strengths (sliders always visible):")
    dpg.add_slider_float(label="Separation Strength", tag=TAG_SLIDER_SEP, default_value=1.5, min_value=0.0, max_value=5.0, callback=slider_sep_cb)
    dpg.add_slider_float(label="Alignment Strength", tag=TAG_SLIDER_ALI, default_value=1.0, min_value=0.0, max_value=5.0, callback=slider_ali_cb)
    dpg.add_slider_float(label="Cohesion Strength", tag=TAG_SLIDER_COH, default_value=1.0, min_value=0.0, max_value=5.0, callback=slider_coh_cb)

    dpg.add_separator()
    dpg.add_checkbox(label="Mouse Enabled (predator)", tag=TAG_CHECK_MOUSE, default_value=False, callback=mouse_toggle_callback)
    dpg.add_spacing(count=1)
    dpg.add_button(label="Start Random Goals", callback=start_goals_callback)
    dpg.add_button(label="Reset Boids", callback=reset_boids_callback)

dpg.setup_dearpygui()
dpg.show_viewport()

# ---------------------------------------------------------------
# PYGAME MAIN LOOP SETUP
# ---------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Simulation with Rule Toggles")

boids = [Boid() for _ in range(NUM_BOIDS)]
clock = pygame.time.Clock()

# Random goal initialization (no goal applied until Start Random Goals pressed)
goal_point = [random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)]
next_goal_time = None
target = None
flee = False
flee_radius = None

# Ensure GUI starts with checkboxes OFF (redundant but explicit)
dpg.set_value(TAG_CHECK_SEP, False)
dpg.set_value(TAG_CHECK_ALI, False)
dpg.set_value(TAG_CHECK_COH, False)
dpg.set_value(TAG_CHECK_MOUSE, False)

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
running = True
while running:
    screen.fill((20,20,20))
    mouse_pos = pygame.mouse.get_pos()

    # Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Decide target logic:
    # Priority: if goals_enabled -> use goal_point
    # otherwise if mouse_enabled and mouse is in window -> use mouse (flee)
    target = None
    flee = False
    flee_radius = None

    if goals_enabled:
        target = goal_point
        flee = False
        # update timed goal
        if next_goal_time is None:
            next_goal_time = time.time() + random.uniform(5,10)
        if time.time() >= next_goal_time:
            goal_point = [random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)]
            next_goal_time = time.time() + random.uniform(5,10)
        # draw goal
        pygame.draw.circle(screen, (0,255,0), (int(goal_point[0]), int(goal_point[1])), 8)
    else:
        # no goals active
        next_goal_time = None

    if not goals_enabled and mouse_enabled:
        # only consider mouse when goals are disabled in this design (keeps intended behavior)
        if 0 <= mouse_pos[0] <= WIDTH and 0 <= mouse_pos[1] <= HEIGHT:
            target = mouse_pos
            flee = True
            flee_radius = PREDATOR_RADIUS
            # draw avoidance radius circle
            pygame.draw.circle(screen, (255,0,0), (int(mouse_pos[0]), int(mouse_pos[1])), PREDATOR_RADIUS, 2)

    # Update and draw boids with rule toggles and slider strengths
    for b in boids:
        neighbors = [o for o in boids if o is not b]
        b.run(
            neighbors,
            target=target,
            flee=flee,
            flee_radius=flee_radius,
            apply_sep=sep_enabled,
            apply_ali=ali_enabled,
            apply_coh=coh_enabled,
            sep_strength=sep_factor,
            ali_strength=ali_factor,
            coh_strength=coh_factor
        )
        b.draw(screen)

    # flip Pygame buffer & tick, then render DearPyGui frame
    pygame.display.flip()
    clock.tick(60)
    dpg.render_dearpygui_frame()

# Cleanup
pygame.quit()
dpg.destroy_context()
