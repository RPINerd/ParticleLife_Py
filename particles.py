"""
Particle Life Simulation using Taichi.

A particle simulation where particles of different colors interact with each other
based on configurable rules within a specified interaction radius.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import taichi as ti

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
NUM_PARTICLE_TYPES = 4
DEFAULT_PARTICLE_COUNT = 800
DEFAULT_WINDOW_SIZE = (1920, 1080)  # Updated window size to 1024x1024
DEFAULT_RADIUS = 50.0
DEFAULT_INTERACTION_STRENGTH = 0.5
DEFAULT_REPULSION_STRENGTH = -0.1
DEFAULT_FRICTION = 0.1
DEFAULT_BORDER_WIDTH = 0.01  # Border width as a fraction of window size
DEFAULT_BORDER_REPULSION = 1.0  # Border repulsion force strength

# Colors in hexadecimal format as expected by Taichi GUI
COLORS = [
    0xEE1010,  # Red
    0x10EE10,  # Green
    0x1010EE,  # Blue
    0xEEEE10  # Yellow
    ]
BACKGROUND_COLOR = 0x131313


@ti.data_oriented
class ParticleLifeSimulation:

    """
    A particle life simulation using Taichi for acceleration.

    This class manages particles of different types that interact based on
    configurable rules within a specific radius.
    """

    def __init__(self,
        particles_per_type: int = DEFAULT_PARTICLE_COUNT,
        window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
        interaction_radius: float = DEFAULT_RADIUS,
        friction: float = DEFAULT_FRICTION,
        border_width: float = DEFAULT_BORDER_WIDTH,
        border_repulsion: float = DEFAULT_BORDER_REPULSION,
        rule_seed: int | None = None) -> None:
        """
        Initialize the particle life simulation.

        Args:
            particles_per_type: Number of particles per type.
            window_size: Size of the simulation window (width, height).
            interaction_radius: Radius within which particles interact.
            friction: Friction coefficient to dampen particle velocities.
            border_width: Width of border repulsion zone as a fraction of window size.
            border_repulsion: Border repulsion force strength.
            rule_seed: Seed for random rule generation.
        """
        # Store configuration
        self.particles_per_type = particles_per_type
        self.total_particles = particles_per_type * NUM_PARTICLE_TYPES
        self.width, self.height = window_size
        self.interaction_radius = interaction_radius
        self.interaction_radius_squared = interaction_radius * interaction_radius
        self.friction = friction
        self.border_width = border_width
        self.border_repulsion = border_repulsion

        # Calculate border distances in normalized coordinates [0-1]
        self.border_x = self.border_width
        self.border_y = self.border_width

        # Set random seed if provided
        if rule_seed is not None:
            random.seed(rule_seed)
            np.random.seed(rule_seed)

        # Create Taichi fields
        # Position vector field (x, y)
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=self.total_particles)
        # Velocity vector field (vx, vy)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=self.total_particles)
        # Particle type field (0-3 for the four colors)
        self.particle_type = ti.field(dtype=ti.i32, shape=self.total_particles)

        # Interaction matrix for rules between different particle types
        self.rule_matrix = ti.field(dtype=ti.f32, shape=(NUM_PARTICLE_TYPES, NUM_PARTICLE_TYPES))

        # Initialize particles and rules
        self._initialize_particles()
        self._initialize_rules()

        # Create window
        window_title = f"Particle Life - {self.total_particles} particles"
        self.gui = ti.GUI(window_title, res=window_size, background_color=BACKGROUND_COLOR)

    def _initialize_particles(self) -> None:
        """
        Initialize particle positions, velocities, and types.
        """
        # Initialize particle positions
        initial_pos = np.zeros((self.total_particles, 2), dtype=np.float32)
        initial_vel = np.zeros((self.total_particles, 2), dtype=np.float32)
        initial_type = np.zeros(self.total_particles, dtype=np.int32)

        # Distribute particles
        for i in range(self.total_particles):
            # Random position within window bounds
            initial_pos[i, 0] = np.random.random()
            initial_pos[i, 1] = np.random.random()

            # Set particle type based on index
            particle_type = i // self.particles_per_type
            initial_type[i] = particle_type

        # Copy to Taichi fields
        self.pos.from_numpy(initial_pos)
        self.vel.from_numpy(initial_vel)
        self.particle_type.from_numpy(initial_type)

        logger.info(f"Initialized {self.total_particles} particles")

    def _initialize_rules(self) -> None:
        """
        Initialize rules for interaction between particle types.
        """
        # Create a rule matrix for interactions
        rule_matrix_np = np.zeros((NUM_PARTICLE_TYPES, NUM_PARTICLE_TYPES), dtype=np.float32)

        # Fill with random values
        for i in range(NUM_PARTICLE_TYPES):
            for j in range(NUM_PARTICLE_TYPES):
                # Attraction/repulsion strength between -0.5 and 1.0
                if i == j:
                    # Same type: slight repulsion
                    rule_matrix_np[i, j] = DEFAULT_REPULSION_STRENGTH
                else:
                    # Different types: random interaction
                    rule_matrix_np[i, j] = (np.random.random() * 1.5) - 0.5

        # Copy to Taichi field
        self.rule_matrix.from_numpy(rule_matrix_np)

        logger.info("Initialized interaction rules")
        logger.debug(f"Rule matrix:\n{rule_matrix_np}")

    @ti.kernel
    def _apply_rules(self, dt: ti.f32):
        """
        Apply interaction rules and update particle positions.

        Args:
            dt: Time step delta.
        """
        # For each particle
        for i in range(self.total_particles):
            # Current particle type and position
            p_type_i = self.particle_type[i]
            p_pos_i = self.pos[i]

            # Reset acceleration
            acceleration = ti.Vector([0.0, 0.0])

            # Calculate border repulsion
            self._apply_border_repulsion(p_pos_i, acceleration)

            # Interact with all other particles
            for j in range(self.total_particles):
                if i != j:  # Don't interact with self
                    # Get other particle's type and position
                    p_type_j = self.particle_type[j]
                    p_pos_j = self.pos[j]

                    # Calculate distance vector
                    delta_pos = p_pos_i - p_pos_j

                    # Scale to window coordinates
                    delta_pos[0] *= self.width
                    delta_pos[1] *= self.height

                    # Calculate squared distance
                    dist_squared = delta_pos[0] * delta_pos[0] + delta_pos[1] * delta_pos[1]

                    # Check if within interaction radius
                    if dist_squared < self.interaction_radius_squared and dist_squared > 1e-6:
                        # Get rule strength between these types
                        strength = self.rule_matrix[p_type_i, p_type_j]

                        # Calculate force (inverse distance)
                        dist = ti.sqrt(dist_squared)
                        force = strength * (1.0 - dist / self.interaction_radius)

                        # Normalize direction vector
                        direction = delta_pos / dist

                        # Apply force
                        acceleration += direction * force

            # Update velocity with acceleration and friction
            self.vel[i] += acceleration * dt
            self.vel[i] *= (1.0 - self.friction)

            # Update position
            self.pos[i] += self.vel[i] * dt

            # Boundary conditions (wrap around)
            for d in ti.static(range(2)):
                if self.pos[i][d] < 0.0:
                    self.pos[i][d] = 1.0 + self.pos[i][d] - ti.floor(self.pos[i][d])
                elif self.pos[i][d] >= 1.0:
                    self.pos[i][d] = self.pos[i][d] - ti.floor(self.pos[i][d])

    @ti.func
    def _apply_border_repulsion(self, pos: ti.template(), acceleration: ti.template()):
        """
        Apply repulsion forces from borders to keep particles away from edges.

        Args:
            pos: Current position of the particle in [0,1] range
            acceleration: Acceleration vector to update
        """
        # Left border repulsion
        if pos[0] < self.border_x:
            # Force increases as the particle gets closer to the edge
            force = self.border_repulsion * (1.0 - pos[0] / self.border_x)
            acceleration[0] += force

        # Right border repulsion
        if pos[0] > (1.0 - self.border_x):
            # Force increases as the particle gets closer to the edge
            force = self.border_repulsion * (1.0 - (1.0 - pos[0]) / self.border_x)
            acceleration[0] -= force

        # Top border repulsion
        if pos[1] < self.border_y:
            # Force increases as the particle gets closer to the edge
            force = self.border_repulsion * (1.0 - pos[1] / self.border_y)
            acceleration[1] += force

        # Bottom border repulsion
        if pos[1] > (1.0 - self.border_y):
            # Force increases as the particle gets closer to the edge
            force = self.border_repulsion * (1.0 - (1.0 - pos[1]) / self.border_y)
            acceleration[1] -= force

    def update(self, dt: float = 0.01) -> bool:
        """
        Update the simulation state and render a frame.

        Args:
            dt: Time step delta.

        Returns:
            bool: False if the simulation should stop, True otherwise.
        """
        # Apply physical rules
        self._apply_rules(dt)

        # Draw particles on the GUI
        particle_positions = self.pos.to_numpy()
        particle_types = self.particle_type.to_numpy()

        # Clear the window
        self.gui.clear(color=BACKGROUND_COLOR)

        # Draw particles by type
        for type_idx in range(NUM_PARTICLE_TYPES):
            # Get indices of particles of this type
            indices = np.where(particle_types == type_idx)[0]
            if len(indices) > 0:
                # Get positions of these particles
                pos_subset = particle_positions[indices]
                # Draw with the color for this type
                self.gui.circles(pos_subset, color=COLORS[type_idx], radius=3)

        # Display the frame and check for events
        self.gui.show()

        # Return False if ESC is pressed or window is closed
        return not self.gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT)

    def save_rules(self, filepath: str) -> None:
        """
        Save the current interaction rules to a JSON file.

        Args:
            filepath: Path to save the rules to.
        """
        rule_matrix = self.rule_matrix.to_numpy()
        rules_dict = {
            "interaction_radius": self.interaction_radius,
            "friction": self.friction,
            "rules": rule_matrix.tolist()
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        with open(filepath, 'w') as f:
            json.dump(rules_dict, f, indent=2)

        logger.info(f"Saved interaction rules to {filepath}")

    def load_rules(self, filepath: str) -> None:
        """
        Load interaction rules from a JSON file.

        Args:
            filepath: Path to load the rules from.
        """
        try:
            with open(filepath) as f:
                rules_dict = json.load(f)

            # Update parameters
            self.interaction_radius = rules_dict.get("interaction_radius", self.interaction_radius)
            self.interaction_radius_squared = self.interaction_radius * self.interaction_radius
            self.friction = rules_dict.get("friction", self.friction)

            # Update rule matrix
            rule_matrix = np.array(rules_dict["rules"], dtype=np.float32)
            self.rule_matrix.from_numpy(rule_matrix)

            logger.info(f"Loaded interaction rules from {filepath}")
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.error(f"Failed to load rules: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Particle Life Simulation using Taichi")

    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLE_COUNT,
                        help=f"Number of particles per type (default: {DEFAULT_PARTICLE_COUNT})")
    parser.add_argument("--width", type=int, default=DEFAULT_WINDOW_SIZE[0],
                        help=f"Window width (default: {DEFAULT_WINDOW_SIZE[0]})")
    parser.add_argument("--height", type=int, default=DEFAULT_WINDOW_SIZE[1],
                        help=f"Window height (default: {DEFAULT_WINDOW_SIZE[1]})")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help=f"Interaction radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION,
                        help=f"Friction coefficient (default: {DEFAULT_FRICTION})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for rule generation (default: None)")
    parser.add_argument("--load-rules", type=str, default=None,
                        help="Path to load interaction rules from")
    parser.add_argument("--save-rules", type=str, default=None,
                        help="Path to save interaction rules to")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode instead of GPU")
    parser.add_argument("--border-width", type=float, default=DEFAULT_BORDER_WIDTH,
                        help=f"Border width as fraction of window size (default: {DEFAULT_BORDER_WIDTH})")
    parser.add_argument("--border-repulsion", type=float, default=DEFAULT_BORDER_REPULSION,
                        help=f"Border repulsion force strength (default: {DEFAULT_BORDER_REPULSION})")

    return parser.parse_args()


def main():
    """
    Main entry point for the particle life simulation.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Taichi - do this only once
    if args.cpu:
        ti.init(arch=ti.cpu)
        logger.info("Using CPU for simulation")
    else:
        try:
            ti.init(arch=ti.opengl)  # Try to use GPU
            logger.info("Using GPU for simulation")
        except Exception:
            ti.init(arch=ti.cpu)  # Fall back to CPU
            logger.info("GPU not available, using CPU for simulation")

    # Create simulation
    simulation = ParticleLifeSimulation(
        particles_per_type=args.particles,
        window_size=(args.width, args.height),
        interaction_radius=args.radius,
        friction=args.friction,
        border_width=args.border_width,
        border_repulsion=args.border_repulsion,
        rule_seed=args.seed
    )

    # Load rules if specified
    if args.load_rules:
        simulation.load_rules(args.load_rules)

    logger.info("Starting simulation - press ESC to exit")

    # Main simulation loop
    running = True
    try:
        while running:
            running = simulation.update(dt=0.01)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error in simulation loop: {e}", exc_info=True)

    # Save rules if specified
    if args.save_rules:
        simulation.save_rules(args.save_rules)

    logger.info("Simulation ended")


if __name__ == "__main__":
    main()
