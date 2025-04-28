"""Core simulation module for Particle Life."""

import logging
import time

import numpy as np
import pygame
import taichi as ti

from .atoms import Atoms

# from .clusters import Clusters
from .settings import Settings
from .ui import UI

logger = logging.getLogger(__name__)


class Simulation:

    """Main simulation class for Particle Life."""

    def __init__(
        self,
        settings: Settings,
        screen: pygame.Surface,
    ) -> None:
        """
        Initialize the simulation.

        Args:
            seed (int): Random seed for consistent results
            width (int): Width of the simulation window
            height (int): Height of the simulation window
            fullscreen (bool): Whether to run in fullscreen mode
            num_colors (int): Number of different colored particles
            atoms_per_color (int): Number of atoms of each color
        """
        self.settings = settings
        self.screen = screen

        # Create main components
        ti.init(arch=ti.opengl)
        self.atoms = Atoms(self.settings)
        # self.clusters = Clusters(self.settings)
        self.ui = UI(self.settings, self._on_settings_changed)

        # Pulse effect parameters
        self.pulse = 0
        self.pulse_x = 0
        self.pulse_y = 0

        # Recording state
        self.recording = False
        self.record_frames = []

        # Performance tracking
        self.last_time = time.time()
        self.total_v = 0.0
        self.exploration_timer = 0

        # Initialize random rules and atoms
        self.random_rules()
        self.reset_atoms()

        logger.info(f"Simulation initialized with seed: {self.settings.seed}")

    def _on_settings_changed(self) -> None:
        """
        Callback for when settings are changed in the UI.

        Updates the simulation accordingly.
        """
        self.atoms.update_count_if_needed()

    def run(self) -> int:
        """
        Run the main simulation loop.

        Returns:
            int: Exit code
        """
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_event(event)

                # Let UI handle events too
                self.ui.process_event(event)

            # Update
            self._update()

            # Draw
            self._draw()

            # Update display
            pygame.display.flip()

            # Update FPS
            self.settings.fps = int(clock.get_fps())
            clock.tick(60)  # Cap to 60fps

        pygame.quit()
        return 0

    def _update(self) -> None:
        """Update simulation state."""
        # Update canvas dimensions if window resized
        width, height = self.screen.get_size()
        if width != self.settings.width or height != self.settings.height:
            self.settings.width = width
            self.settings.height = height

        # Apply physics rules
        start_time = time.time()
        self.total_v = self.atoms.apply_rules(self.pulse, self.pulse_x, self.pulse_y)
        elapsed = time.time() - start_time

        # Log performance occasionally
        if time.time() - self.last_time > 5.0:  # Every 5 seconds
            self.last_time = time.time()
            atoms_count = len(self.atoms.atoms)
            logger.info(f"Physics update for {atoms_count} atoms took {elapsed * 1000:.2f}ms")

        # Update pulse
        if self.pulse != 0:
            self.pulse -= 1 if self.pulse > 0 else -1

        # Handle exploration mode
        if self.settings.explore and self.exploration_timer <= 0:
            self._explore_parameters()
            self.exploration_timer = self.settings.explore_period
        elif self.settings.explore:
            self.exploration_timer -= 1

        # Adapt time scale based on activity
        self._adapt_time_scale()

        # If cluster tracking is enabled, update clusters
        # if self.settings.drawings["clusters"]:
        #     self.clusters.track_clusters(self.atoms.atoms)

        # Record frame if recording
        # if self.recording:
        #     self._record_frame()

    def _draw(self) -> None:
        """Draw all elements to the screen."""
        # Fill background
        self.screen.fill(self.settings.drawings["background_color"])

        # Draw atoms
        self.atoms.draw(self.screen)

        # Draw clusters if enabled
        # if self.settings.drawings["clusters"]:
        #     self.clusters.draw(self.screen)

        # Draw UI
        self.ui.draw()

    def _handle_key_event(self, event: pygame.event.Event) -> None:
        """
        Handle keyboard events.

        Args:
            event (pygame.event.Event): Key event
        """
        if event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.key == pygame.K_r:
            self.random_rules()
        # elif event.key == pygame.K_t:
        #     self.settings.drawings["clusters"] = not self.settings.drawings["clusters"]
        elif event.key == pygame.K_o:
            self.reset_atoms()
        elif event.key == pygame.K_s:
            self.symmetric_rules()
        # elif event.key == pygame.K_f:
        #     self._take_screenshot()
        # elif event.key == pygame.K_v:
        #     self._toggle_recording()

    def _handle_mouse_event(self, event: pygame.event.Event) -> None:
        """
        Handle mouse events.

        Args:
            event (pygame.event.Event): Mouse event
        """
        if event.button == 1:  # Left click
            self.pulse = self.settings.pulse_duration
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.pulse = -self.pulse
            self.pulse_x, self.pulse_y = event.pos

    def random_rules(self) -> None:
        """Generate random rules between particle colors."""
        # Set random seed
        # np.random.seed(self.settings.seed)

        # Initialize rules dictionary
        rules = {}
        radii = {}

        for i in self.settings.colors:
            rules[i] = {}
            for j in self.settings.colors:
                # Random value between -1 and 1
                rules[i][j] = np.random.random() * 2 - 1

            # Set radius
            radii[i] = 80

        # Update settings
        self.settings.rules = rules
        self.settings.radii = radii

        # Flatten rules for efficient computation
        self._flatten_rules()

        logger.info(f"Generated random rules with seed {self.settings.seed}")

    def symmetric_rules(self) -> None:
        """Make interaction rules symmetric between particles."""
        for i in self.settings.colors:
            for j in self.settings.colors:
                if j < i:  # Only need to process each pair once
                    # Average the two values
                    v = 0.5 * (self.settings.rules[i][j] + self.settings.rules[j][i])
                    self.settings.rules[i][j] = self.settings.rules[j][i] = v

        self._flatten_rules()
        self.reset_atoms()
        logger.info("Applied symmetric rules")

    def _flatten_rules(self) -> None:
        """Flatten rules into arrays for efficient computation."""
        self.settings.rules_array = []
        self.settings.radii2_array = []

        for c1 in self.settings.colors:
            for c2 in self.settings.colors:
                self.settings.rules_array.append(self.settings.rules[c1][c2])

            # Square the radius for distance comparisons
            self.settings.radii2_array.append(self.settings.radii[c1] ** 2)

    def reset_atoms(self) -> None:
        """Reset all atoms to random positions."""
        self.atoms.reset()
        # self.clusters.reset()
        logger.info(f"Reset {len(self.atoms.atoms)} atoms to random positions")

    def _explore_parameters(self) -> None:
        """Randomly explore different parameters for the simulation."""
        # Use NumPy for random generation
        rng = np.random.default_rng(int(time.time()))

        # Select a random color
        c1 = self.settings.colors[rng.randint(0, self.settings.num_colors)]

        if rng.random() >= 0.2:  # 80% of the time, change strength
            c2 = self.settings.colors[rng.randint(0, self.settings.num_colors)]
            new_strength = rng.random() * 2 - 1

            # Force opposite-signed values for better results
            if self.settings.rules[c1][c2] > 0:
                new_strength = -new_strength

            self.settings.rules[c1][c2] = new_strength
        else:  # 20% of the time, change radius
            self.settings.radii[c1] = 1 + int(rng.random() * self.settings.max_radius)

        self._flatten_rules()

    def _adapt_time_scale(self) -> None:
        """Adapt time scale based on simulation activity."""
        # Adjust time scale based on velocity
        if self.total_v > 30.0 and self.settings.time_scale > 5.0:
            self.settings.time_scale /= 1.1

        # Gradually normalize time scale
        if self.settings.time_scale < 0.9:
            self.settings.time_scale *= 1.01
        elif self.settings.time_scale > 1.1:
            self.settings.time_scale /= 1.01

    # def _take_screenshot(self) -> None:
    #     """Take a screenshot of the current simulation state."""
    #     screenshot_dir = Path("screenshots")
    #     screenshot_dir.mkdir(exist_ok=True)

    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     filename = f"particle_life_{self.settings.seed}_{timestamp}.png"
    #     filepath = screenshot_dir / filename

    #     pygame.image.save(self.screen, str(filepath))
    #     logger.info(f"Screenshot saved to {filepath}")

    # def _toggle_recording(self) :
    #     """Toggle video recording of the simulation."""
    #     if self.recording:
    #         self._save_recording()
    #         self.recording = False
    #         logger.info("Recording stopped")
    #     else:
    #         self.record_frames = []
    #         self.recording = True
    #         logger.info("Recording started")

    # def _record_frame(self):
    #     """Record the current frame for video output."""
    #     if len(self.record_frames) < 600:  # Limit to 10 seconds at 60fps
    #         self.record_frames.append(pygame.surfarray.array3d(self.screen))

    # def _save_recording(self):
    #     """Save the recorded frames as a video file."""
    #     if not self.record_frames:
    #         logger.warning("No frames to save")
    #         return

    #     try:
    #         import imageio

    #         video_dir = Path("videos")
    #         video_dir.mkdir(exist_ok=True)

    #         timestamp = time.strftime("%Y%m%d_%H%M%S")
    #         filename = f"particle_life_{self.settings.seed}_{timestamp}.mp4"
    #         filepath = video_dir / filename

    #         # Convert frames to uint8 and write video
    #         frames = [frame.transpose(1, 0, 2) for frame in self.record_frames]
    #         imageio.mimsave(
    #             str(filepath),
    #             frames,
    #             fps=60,
    #             quality=8,
    #             codec="h264",
    #         )

    #         logger.info(f"Video saved to {filepath}")
    #         self.record_frames = []
    #     except ImportError:
    #         logger.error("imageio is required for video recording. Install with: pip install imageio imageio-ffmpeg")
