#this is not complete... this code is dangerous for ur laptop/PC - things will get hot!
# boids_optimize.py
# Run: python boids_optimize.py ga
# or:  python boids_optimize.py pso

import math, random, time, sys
from statistics import mean
from copy import deepcopy

# ----------------------------
# Simulation (headless)
# ----------------------------
WIDTH, HEIGHT = 600, 400  # smaller for faster evaluation
MAX_SPEED = 4.0
MAX_FORCE = 0.05

def limit_vec(v, maxv):
    v = list(v)
    mag = math.hypot(v[0], v[1])
    if mag > maxv and mag > 0:
        return [v[0] / mag * maxv, v[1] / mag * maxv]
    return v

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

class Boid:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.pos = [random.uniform(0, self.w), random.uniform(0, self.h)]
        ang = random.uniform(0, 2*math.pi)
        self.vel = [math.cos(ang), math.sin(ang)]
        self.acc = [0.0, 0.0]

    def apply_force(self, f):
        self.acc[0] += f[0]
        self.acc[1] += f[1]

    def update(self):
        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]
        self.vel = limit_vec(self.vel, MAX_SPEED)
        self.pos[0] = (self.pos[0] + self.vel[0]) % self.w
        self.pos[1] = (self.pos[1] + self.vel[1]) % self.h
        self.acc = [0.0, 0.0]

    # steering rules (returns vector)
    def separation(self, boids, desired_sep=25):
        steer = [0.0, 0.0]; count=0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < desired_sep:
                diff = [(self.pos[0]-b.pos[0])/d, (self.pos[1]-b.pos[1])/d]
                steer[0] += diff[0]; steer[1] += diff[1]; count += 1
        if count>0:
            steer[0] /= count; steer[1] /= count
            steer = limit_vec(steer, MAX_FORCE)
        return steer

    def alignment(self, boids, neighbordist=50):
        sumv=[0.0,0.0]; count=0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < neighbordist:
                sumv[0]+=b.vel[0]; sumv[1]+=b.vel[1]; count+=1
        if count>0:
            sumv[0]/=count; sumv[1]/=count
            desired = limit_vec(sumv, MAX_SPEED)
            steer = [desired[0]-self.vel[0], desired[1]-self.vel[1]]
            return limit_vec(steer, MAX_FORCE)
        return [0.0,0.0]

    def cohesion(self, boids, neighbordist=50):
        center=[0.0,0.0]; count=0
        for b in boids:
            d = dist(self.pos, b.pos)
            if 0 < d < neighbordist:
                center[0]+=b.pos[0]; center[1]+=b.pos[1]; count+=1
        if count>0:
            center[0]/=count; center[1]/=count
            desired=[center[0]-self.pos[0], center[1]-self.pos[1]]
            desired = limit_vec(desired, MAX_SPEED)
            steer = [desired[0]-self.vel[0], desired[1]-self.vel[1]]
            return limit_vec(steer, MAX_FORCE)
        return [0.0,0.0]

# headless simulation to evaluate params
def simulate_once(params, steps=250, N=60, seed=None):
    # params: dict with sep, ali, coh strengths (floats)
    if seed is not None:
        random.seed(seed)
    boids = [Boid() for _ in range(N)]
    # warm start randomize a bit
    for _ in range(5):
        for b in boids:
            b.update()

    # accumulators for metrics
    align_scores = []
    coh_scores = []
    sep_penalties = []

    for t in range(steps):
        # compute forces for each boid
        for b in boids:
            others = [o for o in boids if o is not b]
            sep = b.separation(others)
            ali = b.alignment(others)
            coh = b.cohesion(others)

            # apply with given strengths
            b.apply_force([sep[0]*params['sep'], sep[1]*params['sep']])
            b.apply_force([ali[0]*params['ali'], ali[1]*params['ali']])
            b.apply_force([coh[0]*params['coh'], coh[1]*params['coh']])

        # update
        for b in boids:
            b.update()

        # compute metrics this step:
        # alignment: mean cosine similarity to neighborhood mean (range -1..1)
        step_align = []
        centroid = [0.0,0.0]
        for b in boids:
            neighbors = [o for o in boids if o is not b]
            # alignment metric
            meanvx = mean([o.vel[0] for o in neighbors]) if neighbors else 0.0
            meanvy = mean([o.vel[1] for o in neighbors]) if neighbors else 0.0
            # cos sim
            dot = b.vel[0]*meanvx + b.vel[1]*meanvy
            magb = math.hypot(b.vel[0], b.vel[1])
            magm = math.hypot(meanvx, meanvy)
            cos = (dot/(magb*magm)) if (magb>0 and magm>0) else 0.0
            step_align.append(cos)
            centroid[0]+=b.pos[0]; centroid[1]+=b.pos[1]

        align_scores.append(mean(step_align))

        # cohesion: inverse average distance to centroid
        centroid[0]/=len(boids); centroid[1]/=len(boids)
        dists = [dist([centroid[0],centroid[1]], b.pos) for b in boids]
        # smaller average distance => higher cohesion. convert to score by inverse.
        avgd = mean(dists) if dists else 1.0
        coh_scores.append(1.0 / (1.0 + avgd))  # between ~0 and 1

        # separation penalty: count pairs closer than min_sep
        min_sep = 10.0
        pairs = 0
        for i in range(len(boids)):
            for j in range(i+1, len(boids)):
                if dist(boids[i].pos, boids[j].pos) < min_sep:
                    pairs += 1
        # normalize by max possible pairs
        maxpairs = len(boids)*(len(boids)-1)/2
        sep_penalties.append(pairs / maxpairs)

    # aggregate metrics
    align_mean = mean(align_scores)
    coh_mean = mean(coh_scores)
    sep_pen = mean(sep_penalties)

    return {'align': align_mean, 'coh': coh_mean, 'sep_pen': sep_pen}

# fitness function composed from metrics
def evaluate_params(params, runs=3, steps=250, N=60):
    # params: dict with sep, ali, coh
    metrics = []
    for r in range(runs):
        seed = random.randint(0, 2**30)
        m = simulate_once(params, steps=steps, N=N, seed=seed)
        metrics.append(m)
    # average metrics
    align = mean([m['align'] for m in metrics])
    coh   = mean([m['coh'] for m in metrics])
    sep_pen = mean([m['sep_pen'] for m in metrics])

    # Compose fitness: encourage alignment and cohesion, penalize separation collisions
    # weights can be tuned
    fitness = 2.0 * align + 1.5 * coh - 2.5 * sep_pen
    # we return fitness and component breakdown
    return fitness, {'align': align, 'coh': coh, 'sep_pen': sep_pen}

# ----------------------------
# Genetic Algorithm (simple)
# ----------------------------
def random_individual(bounds):
    # bounds: list of (min,max) per parameter
    return [random.uniform(a,b) for (a,b) in bounds]

def mutate_ga(ind, bounds, mutprob=0.2, sigma=0.3):
    new = ind[:]
    for i in range(len(new)):
        if random.random() < mutprob:
            new[i] += random.gauss(0, sigma)
            # clamp
            a,b = bounds[i]
            new[i] = max(a, min(b, new[i]))
    return new

def crossover_blend(a, b, alpha=0.5):
    # BLX / blend - produce child near parents
    child = []
    for i in range(len(a)):
        lo = min(a[i], b[i]) - alpha * abs(a[i]-b[i])
        hi = max(a[i], b[i]) + alpha * abs(a[i]-b[i])
        child.append(random.uniform(lo, hi))
    return child

def run_ga(generations=30, popsize=24, elitism=2, runs_per_eval=2):
    # optimize three params: sep, ali, coh; bounds:
    bounds = [(0.0,5.0), (0.0,5.0), (0.0,5.0)]
    # population
    pop = [random_individual(bounds) for _ in range(popsize)]
    fitness_cache = {}
    best_overall = None
    for gen in range(generations):
        scored = []
        for ind in pop:
            key = tuple(round(x,6) for x in ind)
            if key in fitness_cache:
                fit = fitness_cache[key]
            else:
                params = {'sep': ind[0], 'ali': ind[1], 'coh': ind[2]}
                fit, comps = evaluate_params(params, runs=runs_per_eval)
                fitness_cache[key] = (fit, comps)
            scored.append((ind, fitness_cache[key][0]))
        # sort desc
        scored.sort(key=lambda x: x[1], reverse=True)
        if best_overall is None or scored[0][1] > best_overall[0]:
            best_overall = (scored[0][1], scored[0][0])
        print(f"[GA] Gen {gen:02d} best fitness {scored[0][1]:.4f} params {scored[0][0]}")
        # selection: keep elites
        newpop = [deepcopy(x[0]) for x in scored[:elitism]]
        # tournament selection helper
        def tour_select():
            a = random.choice(pop)
            b = random.choice(pop)
            ka = fitness_cache[tuple(round(x,6) for x in a)][0]
            kb = fitness_cache[tuple(round(x,6) for x in b)][0]
            return a if ka > kb else b
        # fill rest with crossover + mutation
        while len(newpop) < popsize:
            p1 = tour_select()
            p2 = tour_select()
            child = crossover_blend(p1, p2, alpha=0.2)
            child = mutate_ga(child, bounds, mutprob=0.3, sigma=0.4)
            # clamp to bounds
            for i,(lo,hi) in enumerate(bounds):
                child[i] = max(lo, min(hi, child[i]))
            newpop.append(child)
        pop = newpop
    # final best
    print("[GA] Best overall:", best_overall)
    return best_overall

# ----------------------------
# PSO (simple)
# ----------------------------
def run_pso(iterations=60, swarm_size=24):
    dim = 3
    bounds = [(0.0,5.0)]*dim
    # PSO hyperparams
    w_inertia = 0.7
    c1 = 1.4
    c2 = 1.4
    # initialize particles
    particles = []
    for _ in range(swarm_size):
        pos = [random.uniform(a,b) for (a,b) in bounds]
        vel = [random.uniform(-1,1) for _ in range(dim)]
        particles.append({'pos': pos, 'vel': vel, 'best_pos': pos[:], 'best_fit': -1e9})
    gbest_pos = None
    gbest_fit = -1e9

    for it in range(iterations):
        for p in particles:
            key = tuple(round(x,6) for x in p['pos'])
            # evaluate
            params = {'sep': p['pos'][0], 'ali': p['pos'][1], 'coh': p['pos'][2]}
            fit, comps = evaluate_params(params, runs=2)
            # update personal best
            if fit > p['best_fit']:
                p['best_fit'] = fit
                p['best_pos'] = p['pos'][:]
            # update global
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p['pos'][:]
        # update velocities and positions
        for p in particles:
            for d in range(dim):
                r1 = random.random(); r2 = random.random()
                cognitive = c1 * r1 * (p['best_pos'][d] - p['pos'][d])
                social = c2 * r2 * (gbest_pos[d] - p['pos'][d])
                p['vel'][d] = w_inertia * p['vel'][d] + cognitive + social
                # clamp velocity
                maxv = (bounds[d][1] - bounds[d][0]) * 0.2
                p['vel'][d] = max(-maxv, min(maxv, p['vel'][d]))
                # step
                p['pos'][d] += p['vel'][d]
                # clamp to bounds
                p['pos'][d] = max(bounds[d][0], min(bounds[d][1], p['pos'][d]))
        if it % 5 == 0 or it == iterations-1:
            print(f"[PSO] Iter {it:03d} gbest {gbest_fit:.4f} pos {gbest_pos}")
    print("[PSO] Best overall:", (gbest_fit, gbest_pos))
    return gbest_fit, gbest_pos

# ----------------------------
# CLI driver
# ----------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ga"
    random.seed(12345)
    t0 = time.time()
    if mode.lower() in ("ga","genetic"):
        best = run_ga(generations=20, popsize=18, elitism=2, runs_per_eval=2)
        print("GA result:", best)
    elif mode.lower() in ("pso",):
        best = run_pso(iterations=40, swarm_size=18)
        print("PSO result:", best)
    else:
        print("Usage: python boids_optimize.py [ga|pso]")
    print("Elapsed:", time.time() - t0)
