"""
Atoms module for Particle Life simulation.

This module handles the creation, management, and physics of atoms/particles.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pygame
import pyopencl as cl

from .opencl_utils import PARTICLE_KERNEL, CLProgram, check_opencl_available, get_opencl_context

if TYPE_CHECKING:
    from particlelife.settings import Settings

logger = logging.getLogger(__name__)


class Atoms:

    """Manages atoms/particles and their interactions in the simulation."""

    def __init__(self, settings: "Settings") -> None:
        """
        Initialize the atoms system.

        Args:
            settings (Settings): Simulation settings
        """
        self.settings = settings
        self.atoms = np.zeros((0, 5))  # [x, y, vx, vy, color_idx]
        self.prev_atom_count = 0

        # Initialize OpenCL if requested
        if self.settings.use_gpu:
            self._initialize_opencl()

    def _initialize_opencl(self) -> None:
        """Initialize OpenCL context, queue, and kernels."""
        # First check if OpenCL is available
        available, error_msg = check_opencl_available()
        if not available:
            logger.error(f"OpenCL initialization failed: {error_msg}")
            logger.error("Falling back to CPU computation")
            logger.error("To use GPU acceleration, install appropriate OpenCL drivers for your hardware:")
            logger.error("- For NVIDIA GPUs: Install NVIDIA drivers and CUDA toolkit")
            logger.error("- For AMD GPUs: Install AMD drivers with OpenCL support")
            logger.error("- For Intel GPUs: Install Intel OpenCL runtime")
            self.settings.use_gpu = False
            return

        try:
            # Get the OpenCL context, device, and queue
            context, device, queue = get_opencl_context(
                platform_index=getattr(self.settings, 'platform_index', 0),
                device_index=getattr(self.settings, 'device_index', 0)
            )

            # Store in settings for potential reuse
            self.settings.opencl_context = context
            self.settings.opencl_device = device
            self.settings.opencl_queue = queue

            # Create program with the particle simulation kernel
            program = CLProgram(context, PARTICLE_KERNEL)
            self.settings.opencl_program = program
            self.settings.opencl_kernel = program.get_kernel("apply_forces")

            self.settings.opencl_initialized = True
            logger.info("OpenCL initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            logger.error("Falling back to CPU computation")
            self.settings.use_gpu = False

    def reset(self) -> None:
        """Reset atoms to random positions with zero velocity."""
        count = self.settings.atoms_settings["count"]
        num_colors = self.settings.num_colors

        # Initialize with zeros and then fill with random values
        total_atoms = count * num_colors
        self.atoms = np.zeros((total_atoms, 5))

        # Generate random positions
        rng = np.random.default_rng()

        # Apply random positions
        width = self.settings.width
        height = self.settings.height
        self.atoms[:, 0] = rng.random(total_atoms) * (width - 100) + 50
        self.atoms[:, 1] = rng.random(total_atoms) * (height - 100) + 50

        # Set colors
        for i in range(num_colors):
            start_idx = i * count
            end_idx = (i + 1) * count
            self.atoms[start_idx:end_idx, 4] = i

        self.prev_atom_count = count

        # Reinitialize OpenCL buffers if using GPU
        if self.settings.use_gpu and self.settings.opencl_initialized:
            self._create_opencl_buffers()

    def _create_opencl_buffers(self) -> None:
        """Create and populate OpenCL buffers for particle simulation."""
        if not self.settings.opencl_initialized:
            return

        context = self.settings.opencl_context
        queue = self.settings.opencl_queue

        # Extract positions, velocities, and colors from the atoms array
        positions_velocities = np.zeros((len(self.atoms), 4), dtype=np.float32)
        positions_velocities[:, 0] = self.atoms[:, 0]  # x
        positions_velocities[:, 1] = self.atoms[:, 1]  # y
        positions_velocities[:, 2] = self.atoms[:, 2]  # vx
        positions_velocities[:, 3] = self.atoms[:, 3]  # vy

        colors = np.array(self.atoms[:, 4], dtype=np.int32)

        # Convert rules and radii to NumPy arrays
        rules_array = np.array(self.settings.rules_array, dtype=np.float32)
        radii2_array = np.array(self.settings.radii2_array, dtype=np.float32)

        # Create output array for total velocities
        total_velocities = np.zeros(len(self.atoms), dtype=np.float32)

        # Create OpenCL buffers
        mf = cl.mem_flags
        self.settings.cl_positions_velocities = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions_velocities
        )
        self.settings.cl_colors = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=colors
        )
        self.settings.cl_rules_array = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_array
        )
        self.settings.cl_radii2_array = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=radii2_array
        )
        self.settings.cl_total_velocities = cl.Buffer(
            context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=total_velocities
        )

    def update_count_if_needed(self) -> None:
        """
        Check if the atom count has changed and update accordingly.

        This ensures the atoms array matches the current settings.
        """
        current_count = self.settings.atoms_settings["count"]
        if current_count != self.prev_atom_count:
            logger.info(f"Atoms count changed from {self.prev_atom_count} to {current_count}")
            self.reset()

    def apply_rules(self, pulse: int, pulse_x: float, pulse_y: float) -> float:
        """
        Apply the interaction rules between atoms.

        Args:
            pulse (int): Current pulse strength
            pulse_x (float): X coordinate of the pulse center
            pulse_y (float): Y coordinate of the pulse center

        Returns:
            float: Total velocity (measure of activity)
        """
        if len(self.atoms) == 0:
            return 0.0

        # Use GPU version if enabled
        if self.settings.use_gpu and self.settings.opencl_initialized:
            return self._apply_rules_gpu(pulse, pulse_x, pulse_y)
        return self._apply_rules_cpu(pulse, pulse_x, pulse_y)

    def _apply_rules_gpu(self, pulse: int, pulse_x: float, pulse_y: float) -> float:
        """
        Apply the interaction rules between atoms using GPU acceleration.

        Args:
            pulse (int): Current pulse strength
            pulse_x (float): X coordinate of the pulse center
            pulse_y (float): Y coordinate of the pulse center

        Returns:
            float: Total velocity (measure of activity)
        """
        queue = self.settings.opencl_queue
        kernel = self.settings.opencl_kernel

        # For performance, extract settings values used in calculations
        width = np.float32(self.settings.width)
        height = np.float32(self.settings.height)
        time_scale = np.float32(self.settings.time_scale)
        viscosity = np.float32(self.settings.viscosity)
        wall_repel = np.float32(self.settings.wall_repel)
        gravity = np.float32(self.settings.gravity)
        num_atoms = np.int32(len(self.atoms))
        num_colors = np.int32(self.settings.num_colors)

        # Check if rules have changed, and update buffer if needed
        rules_array = np.array(self.settings.rules_array, dtype=np.float32)
        cl.enqueue_copy(queue, self.settings.cl_rules_array, rules_array)

        # Check if radii have changed, and update buffer if needed
        radii2_array = np.array(self.settings.radii2_array, dtype=np.float32)
        cl.enqueue_copy(queue, self.settings.cl_radii2_array, radii2_array)

        # Run the kernel
        global_size = (len(self.atoms),)
        local_size = None  # Let OpenCL choose the local size

        kernel(
            queue, global_size, local_size,
            self.settings.cl_positions_velocities,
            self.settings.cl_colors,
            self.settings.cl_rules_array,
            self.settings.cl_radii2_array,
            num_atoms,
            num_colors,
            width,
            height,
            time_scale,
            viscosity,
            wall_repel,
            gravity,
            np.int32(pulse),
            np.float32(pulse_x),
            np.float32(pulse_y),
            self.settings.cl_total_velocities
        )

        # Read back the positions and velocities
        positions_velocities = np.empty((len(self.atoms), 4), dtype=np.float32)
        cl.enqueue_copy(queue, positions_velocities, self.settings.cl_positions_velocities)

        # Read back total velocities
        total_velocities = np.empty(len(self.atoms), dtype=np.float32)
        cl.enqueue_copy(queue, total_velocities, self.settings.cl_total_velocities)

        # Update the atoms array
        self.atoms[:, 0] = positions_velocities[:, 0]  # x
        self.atoms[:, 1] = positions_velocities[:, 1]  # y
        self.atoms[:, 2] = positions_velocities[:, 2]  # vx
        self.atoms[:, 3] = positions_velocities[:, 3]  # vy

        # Calculate total velocity
        total_v = np.sum(total_velocities) / len(self.atoms)

        return total_v

    def _apply_rules_cpu(self, pulse: int, pulse_x: float, pulse_y: float) -> float:
        """
        Apply the interaction rules between atoms using CPU.

        Args:
            pulse (int): Current pulse strength
            pulse_x (float): X coordinate of the pulse center
            pulse_y (float): Y coordinate of the pulse center

        Returns:
            float: Total velocity (measure of activity)
        """
        # For performance, extract settings values used in calculations
        width = self.settings.width
        height = self.settings.height
        time_scale = self.settings.time_scale
        viscosity = self.settings.viscosity
        wall_repel = self.settings.wall_repel
        gravity = self.settings.gravity

        # Calculate forces efficiently using vectorized operations
        total_atoms = len(self.atoms)
        forces = np.zeros((total_atoms, 2))  # [fx, fy]

        # Create a view of just the positions and colors for cleaner code
        positions = self.atoms[:, 0:2]
        colors = self.atoms[:, 4].astype(np.int32)

        # Calculate pairwise distances and forces
        for i in range(total_atoms):
            # Get the rules index based on color
            idx = int(colors[i]) * self.settings.num_colors
            color_radius2 = self.settings.radii2_array[int(colors[i])]

            # Calculate distances to all other atoms in one operation
            diffs = positions[i] - positions
            distances2 = np.sum(diffs**2, axis=1)

            # Create a mask to skip self-interaction (distance == 0)
            mask = (distances2 > 0) & (distances2 < color_radius2)

            if np.any(mask):
                # Get the relevant rules for each atom
                rule_indices = idx + colors[mask]
                interaction_rules = np.array([self.settings.rules_array[int(ri)] for ri in rule_indices])

                # Calculate distances and normalize directions
                distances = np.sqrt(distances2[mask])
                normalized_diffs = diffs[mask] / distances[:, np.newaxis]

                # Calculate forces based on rules and distances
                force_magnitudes = interaction_rules / distances
                force_vectors = normalized_diffs * force_magnitudes[:, np.newaxis]

                # Accumulate forces
                forces[i] += np.sum(force_vectors, axis=0)

                # Draw lines between interacting atoms if enabled
                if self.settings.drawings["lines"]:
                    pass  # Would store these if needed for drawing

            # Apply pulse if active
            if pulse != 0:
                dx = self.atoms[i, 0] - pulse_x
                dy = self.atoms[i, 1] - pulse_y
                d2 = dx * dx + dy * dy
                if d2 > 0:
                    pulse_force = 100.0 * pulse / (d2 * time_scale)
                    forces[i, 0] += pulse_force * dx
                    forces[i, 1] += pulse_force * dy

            # Apply wall repulsion if enabled
            if wall_repel > 0:
                if self.atoms[i, 0] < wall_repel:
                    forces[i, 0] += (wall_repel - self.atoms[i, 0]) * 0.1
                if self.atoms[i, 0] > width - wall_repel:
                    forces[i, 0] += (width - wall_repel - self.atoms[i, 0]) * 0.1
                if self.atoms[i, 1] < wall_repel:
                    forces[i, 1] += (wall_repel - self.atoms[i, 1]) * 0.1
                if self.atoms[i, 1] > height - wall_repel:
                    forces[i, 1] += (height - wall_repel - self.atoms[i, 1]) * 0.1

            # Apply gravity
            forces[i, 1] += gravity

        # Update velocities with forces and apply viscosity
        vmix = 1.0 - viscosity
        self.atoms[:, 2] = self.atoms[:, 2] * vmix + forces[:, 0] * time_scale
        self.atoms[:, 3] = self.atoms[:, 3] * vmix + forces[:, 1] * time_scale

        # Calculate total velocity for time scaling
        total_v = np.sum(np.abs(self.atoms[:, 2:4]))
        total_v /= total_atoms

        # Update positions
        self.atoms[:, 0:2] += self.atoms[:, 2:4]

        # Boundary conditions - bounce off walls
        # X boundaries
        x_too_low = self.atoms[:, 0] < 0
        if np.any(x_too_low):
            self.atoms[x_too_low, 0] = -self.atoms[x_too_low, 0]
            self.atoms[x_too_low, 2] *= -1

        x_too_high = self.atoms[:, 0] >= width
        if np.any(x_too_high):
            self.atoms[x_too_high, 0] = 2 * width - self.atoms[x_too_high, 0]
            self.atoms[x_too_high, 2] *= -1

        # Y boundaries
        y_too_low = self.atoms[:, 1] < 0
        if np.any(y_too_low):
            self.atoms[y_too_low, 1] = -self.atoms[y_too_low, 1]
            self.atoms[y_too_low, 3] *= -1

        y_too_high = self.atoms[:, 1] >= height
        if np.any(y_too_high):
            self.atoms[y_too_high, 1] = 2 * height - self.atoms[y_too_high, 1]
            self.atoms[y_too_high, 3] *= -1

        return total_v

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw atoms to the provided surface.

        Args:
            surface (pygame.Surface): Surface to draw on
        """
        radius = self.settings.atoms_settings["radius"]

        for i in range(len(self.atoms)):
            x, y = int(self.atoms[i, 0]), int(self.atoms[i, 1])
            color_idx = int(self.atoms[i, 4])
            color = self.settings.get_color_rgb(color_idx)

            if self.settings.drawings["circle"]:
                pygame.draw.circle(surface, color, (x, y), radius)
            else:
                # Draw as square
                rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
                pygame.draw.rect(surface, color, rect)
