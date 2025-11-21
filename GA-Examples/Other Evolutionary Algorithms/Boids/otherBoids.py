#older implementation
import pygame
import numpy as np
import math

class Boid:
    def __init__(self, x, y, max_speed=4, max_force=0.15):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)], dtype=float)
        self.acc = np.array([0.0, 0.0], dtype=float)
        self.max_speed = max_speed
        self.max_force = max_force
    
    def apply_force(self, force):
        self.acc += force
    
    def separation(self, boids, desired_sep=40):
        """Steer to avoid crowding local flockmates"""
        steer = np.array([0.0, 0.0])
        count = 0
        for other in boids:
            dist = np.linalg.norm(self.pos - other.pos)
            if 0 < dist < desired_sep:
                direction = self.pos - other.pos
                direction = direction / (dist + 1e-6)
                steer += direction
                count += 1
        
        if count > 0:
            steer = steer / count
            steer = self._limit_force(steer)
        return steer
    
    def alignment(self, boids, perception_radius=60):
        """Steer towards average heading of local flockmates"""
        avg_vel = np.array([0.0, 0.0])
        count = 0
        for other in boids:
            dist = np.linalg.norm(self.pos - other.pos)
            if 0 < dist < perception_radius:
                avg_vel += other.vel
                count += 1
        
        if count > 0:
            avg_vel = avg_vel / count
            avg_vel = (avg_vel / np.linalg.norm(avg_vel + 1e-6)) * self.max_speed
            steer = avg_vel - self.vel
            steer = self._limit_force(steer)
            return steer
        return np.array([0.0, 0.0])
    
    def cohesion(self, boids, perception_radius=60):
        """Steer to move toward average location of local flockmates"""
        center_mass = np.array([0.0, 0.0])
        count = 0
        for other in boids:
            dist = np.linalg.norm(self.pos - other.pos)
            if 0 < dist < perception_radius:
                center_mass += other.pos
                count += 1
        
        if count > 0:
            center_mass = center_mass / count
            desired = center_mass - self.pos
            dist = np.linalg.norm(desired)
            if dist > 0:
                desired = (desired / dist) * self.max_speed
                steer = desired - self.vel
                steer = self._limit_force(steer)
                return steer
        return np.array([0.0, 0.0])
    
    def seek(self, target):
        """Steer towards a target"""
        desired = target - self.pos
        dist = np.linalg.norm(desired)
        if dist > 0:
            desired = (desired / dist) * self.max_speed
            steer = desired - self.vel
            return self._limit_force(steer)
        return np.array([0.0, 0.0])
    
    def flee(self, target):
        """Steer away from a target"""
        desired = self.pos - target
        dist = np.linalg.norm(desired)
        if dist > 0:
            desired = (desired / dist) * self.max_speed
            steer = desired - self.vel
            return self._limit_force(steer * 2)  # Flee is more urgent
        return np.array([0.0, 0.0])
    
    def _limit_force(self, force):
        """Limit force magnitude"""
        mag = np.linalg.norm(force)
        if mag > self.max_force:
            return (force / mag) * self.max_force
        return force
    
    def update(self):
        """Update position based on accumulated forces"""
        self.vel += self.acc
        
        # Limit speed
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        
        self.pos += self.vel
        
        # Wrap around edges
        if self.pos[0] > 1200:
            self.pos[0] = 0
        elif self.pos[0] < 0:
            self.pos[0] = 1200
        
        if self.pos[1] > 700:
            self.pos[1] = 0
        elif self.pos[1] < 0:
            self.pos[1] = 700
        
        self.acc = np.array([0.0, 0.0])
    
    def draw(self, screen):
        """Draw boid as triangle pointing in direction of velocity"""
        angle = math.atan2(self.vel[1], self.vel[0])
        size = 6
        
        x, y = self.pos
        points = [
            (x + math.cos(angle) * size, y + math.sin(angle) * size),
            (x + math.cos(angle + 2.5) * size, y + math.sin(angle + 2.5) * size),
            (x + math.cos(angle - 2.5) * size, y + math.sin(angle - 2.5) * size)
        ]
        pygame.draw.polygon(screen, (255, 255, 255), points)

def main():
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    
    # Create boids
    boids = [Boid(np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT)) 
             for _ in range(40)]
    
    # GUI parameters
    sep_weight = 1.5
    ali_weight = 1.0
    coh_weight = 1.0
    is_food = True
    predator_radius = 150
    
    running = True
    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_food = not is_food
        
        # Keyboard controls for weights
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            sep_weight = min(5.0, sep_weight + 0.03)
        if keys[pygame.K_DOWN]:
            sep_weight = max(0.0, sep_weight - 0.03)
        
        if keys[pygame.K_w]:
            ali_weight = min(5.0, ali_weight + 0.03)
        if keys[pygame.K_s]:
            ali_weight = max(0.0, ali_weight - 0.03)
        
        if keys[pygame.K_a]:
            coh_weight = min(5.0, coh_weight + 0.03)
        if keys[pygame.K_d]:
            coh_weight = max(0.0, coh_weight - 0.03)
        
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
        
        # Update boids
        for boid in boids:
            sep = boid.separation(boids) * sep_weight
            ali = boid.alignment(boids) * ali_weight
            coh = boid.cohesion(boids) * coh_weight
            
            boid.apply_force(sep)
            boid.apply_force(ali)
            boid.apply_force(coh)
            
            # Mouse interaction
            dist_to_mouse = np.linalg.norm(boid.pos - mouse_pos)
            if dist_to_mouse < predator_radius:
                if is_food:
                    boid.apply_force(boid.seek(mouse_pos) * 0.5)
                else:
                    boid.apply_force(boid.flee(mouse_pos) * 0.8)
            
            boid.update()
        
        # Draw
        screen.fill((20, 20, 20))
        
        # Draw mouse indicator
        mouse_color = (0, 255, 0) if is_food else (255, 0, 0)
        pygame.draw.circle(screen, mouse_color, (int(mouse_pos[0]), int(mouse_pos[1])), 10, 2)
        pygame.draw.circle(screen, mouse_color, (int(mouse_pos[0]), int(mouse_pos[1])), predator_radius, 1)
        
        # Draw boids
        for boid in boids:
            boid.draw(screen)
        
        # Draw UI text
        sep_text = font.render(f"SEP: {sep_weight:.2f} (UP/DOWN)", True, (200, 200, 200))
        ali_text = font.render(f"ALI: {ali_weight:.2f} (W/S)", True, (200, 200, 200))
        coh_text = font.render(f"COH: {coh_weight:.2f} (A/D)", True, (200, 200, 200))
        mode_text = font.render(f"MODE: {'FOOD (green)' if is_food else 'PREDATOR (red)'} - SPACE to toggle", True, mouse_color)
        
        screen.blit(sep_text, (10, 10))
        screen.blit(ali_text, (10, 35))
        screen.blit(coh_text, (10, 60))
        screen.blit(mode_text, (10, 90))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()