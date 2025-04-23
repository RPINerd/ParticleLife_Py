"""
Clusters module for Particle Life simulation.

Handles detection and visualization of particle clusters.
"""

import logging

import numpy as np
import pygame

logger = logging.getLogger(__name__)


class Clusters:

    """Manages clustering of atoms for visualization and tracking."""

    def __init__(self, settings):
        """
        Initialize clusters tracker.

        Args:
            settings (Settings): Simulation settings
        """
        self.settings = settings
        self.clusters = []  # [x, y, radius, color_idx]

    def reset(self):
        """Clear all tracked clusters."""
        self.clusters = []

    def track_clusters(self, atoms):
        """
        Track and update clusters of atoms.

        Args:
            atoms (numpy.ndarray): Array of atom data [x, y, vx, vy, color_idx]
        """
        # Add new clusters if we have too few
        self._add_new_clusters()

        if not self.clusters or len(atoms) == 0:
            return

        # Convert to numpy array for faster operations
        clusters_arr = np.array(self.clusters)

        # Initialize accumulators for each cluster
        # [count, sum_x, sum_y, sum_d2, sum_color]
        accumulators = np.zeros((len(self.clusters), 5))

        # Maximum K-means passes for convergence
        max_k_mean_passes = 10

        for _ in range(max_k_mean_passes):
            # Reset accumulators
            accumulators.fill(0)

            # For each atom, find the closest cluster
            for atom in atoms:
                # Calculate distances to all clusters
                dx = clusters_arr[:, 0] - atom[0]
                dy = clusters_arr[:, 1] - atom[1]
                distances2 = dx**2 + dy**2

                # Find the closest cluster within max radius
                min_idx = np.argmin(distances2)
                min_dist2 = distances2[min_idx]

                if min_dist2 < self.settings.max_radius**2:
                    # Accumulate data for the closest cluster
                    accumulators[min_idx, 0] += 1
                    accumulators[min_idx, 1] += atom[0]
                    accumulators[min_idx, 2] += atom[1]
                    accumulators[min_idx, 3] += min_dist2
                    accumulators[min_idx, 4] += atom[4]

            # Move clusters to the center of their assigned atoms
            max_displacement = 0.0
            for i in range(len(self.clusters)):
                if accumulators[i, 0] > self.settings.min_cluster_size:
                    new_x = accumulators[i, 1] / accumulators[i, 0]
                    new_y = accumulators[i, 2] / accumulators[i, 0]

                    # Track maximum displacement for convergence check
                    max_displacement = max(
                        max_displacement,
                        abs(clusters_arr[i, 0] - new_x),
                        abs(clusters_arr[i, 1] - new_y)
                    )

                    # Update cluster position
                    clusters_arr[i, 0] = new_x
                    clusters_arr[i, 1] = new_y

            # If clusters have moved very little, we've converged
            if max_displacement < 1.0:
                break

        # Finalize clusters (update radius and color)
        self._finalize_clusters(clusters_arr, accumulators)

    def _add_new_clusters(self):
        """Add new clusters if needed to maintain desired count."""
        if len(self.clusters) < self.settings.max_clusters // 2:
            rng = np.random.RandomState()

            # Add new clusters until we reach max_clusters
            while len(self.clusters) < self.settings.max_clusters:
                x = rng.random() * (self.settings.width - 100) + 50
                y = rng.random() * (self.settings.height - 100) + 50

                # [x, y, radius, color_idx]
                self.clusters.append([x, y, self.settings.max_radius, 0])

    def _finalize_clusters(self, clusters_arr, accumulators):
        """
        Finalize cluster properties based on accumulated data.

        Args:
            clusters_arr (numpy.ndarray): Array of cluster data
            accumulators (numpy.ndarray): Accumulated data for each cluster
        """
        # Update self.clusters with new data
        self.clusters = []

        for i in range(len(clusters_arr)):
            count = accumulators[i, 0]

            if count > self.settings.min_cluster_size:
                # Normalize
                norm = 1.0 / count

                # Calculate new radius with 10% extra room
                new_radius = 1.10 * np.sqrt(accumulators[i, 3] * norm)

                # Smoothly update radius (exponential smoothing)
                smoothed_radius = 0.95 * clusters_arr[i, 2] + 0.05 * new_radius

                # Determine average color
                avg_color = int(round(accumulators[i, 4] * norm))

                # Add to our list
                self.clusters.append([
                    clusters_arr[i, 0],
                    clusters_arr[i, 1],
                    smoothed_radius,
                    avg_color
                ])
            else:
                # Add disabled cluster (radius 0)
                self.clusters.append([
                    clusters_arr[i, 0],
                    clusters_arr[i, 1],
                    0.0,
                    0
                ])

    def draw(self, surface):
        """
        Draw clusters to the surface.

        Args:
            surface (pygame.Surface): Surface to draw on
        """
        i = 0
        while i < len(self.clusters):
            x, y, radius, color_idx = self.clusters[i]

            if radius > 0:
                # Draw unfilled circle
                color = self.settings.get_color_rgb(color_idx)
                pygame.draw.circle(surface, color, (int(x), int(y)), int(radius), 1)
                i += 1
            else:
                # Remove disabled clusters
                self.clusters.pop(i)
