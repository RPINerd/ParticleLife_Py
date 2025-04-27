"""Settings module for Particle Life simulation."""


class Settings:

    """Stores and manages settings for the Particle Life simulation."""

    def __init__(
        self,
        seed: int = 4294967294,
        width: int = 800,
        height: int = 600,
        num_colors: int = 4,
        atoms_per_color: int = 500,
        use_gpu: bool = False,
    ) -> None:
        """
        Initialize simulation settings.

        Args:
            seed (int): Random seed for consistent results
            width (int): Width of the simulation window
            height (int): Height of the simulation window
            num_colors (int): Number of different colored particles
            atoms_per_color (int): Number of atoms of each color
            use_gpu (bool): Whether to use GPU acceleration
        """
        # Display settings
        self.width = width
        self.height = height

        # Simulation constants
        self.max_radius = 200
        self.max_clusters = 20
        self.min_cluster_size = 50
        self.predefined_colors = [
            (0, 255, 0),     # green
            (255, 0, 0),     # red
            (255, 165, 0),   # orange
            (0, 255, 255),   # cyan
            (255, 0, 255),   # magenta
            (230, 230, 250),  # lavender
            (0, 128, 128),   # teal
        ]

        # Simulation settings
        self.seed = seed
        self.fps = 0
        self.num_colors = min(num_colors, len(self.predefined_colors))
        self.time_scale = 1.0
        self.viscosity = 0.7  # speed dampening
        self.gravity = 0.0
        self.pulse_duration = 10
        self.wall_repel = 40
        self.explore = False
        self.explore_period = 100

        # GPU acceleration settings
        self.use_gpu = use_gpu
        self.opencl_initialized = False
        self.opencl_context = None
        self.opencl_queue = None
        self.opencl_program = None
        self.opencl_kernel = None
        self.opencl_device = None

        # Buffers for OpenCL
        self.cl_positions_velocities = None
        self.cl_colors = None
        self.cl_rules_array = None
        self.cl_radii2_array = None
        self.cl_total_velocities = None

        # Colors list - populated based on num_colors
        self.colors = self._get_colors()

        # Atom settings
        self.atoms_settings = {
            "count": atoms_per_color,
            "radius": 1,
        }

        # Drawing settings
        self.drawings = {
            "lines": False,
            "circle": False,
            "clusters": False,
            "background_color": (0, 0, 0),  # black
        }

        # Rules - will be populated with random values
        self.rules = {}
        self.rules_array = []
        self.radii = {}
        self.radii2_array = []

    def _get_colors(self) -> list[str]:
        """Get the list of colors based on num_colors setting."""
        colors = []
        for i in range(self.num_colors):
            # Store the color index as a string key for dictionary access
            colors.append(str(i))
        return colors

    def get_color_rgb(self, color_idx: int | str) -> tuple[int, int, int]:
        """
        Get RGB tuple for a color index.

        Args:
            color_idx (int): Index of the color

        Returns:
            tuple: RGB tuple
        """
        if isinstance(color_idx, str):
            color_idx = int(color_idx)
        return self.predefined_colors[color_idx]
