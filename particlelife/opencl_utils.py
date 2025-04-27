"""OpenCL utilities for Particle Life simulation."""

import logging
from typing import Any

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except (ImportError, Exception) as e:
    # Handle both import errors and initialization errors
    cl = None
    OPENCL_AVAILABLE = False
    init_error = str(e)

logger = logging.getLogger(__name__)


def check_opencl_available() -> tuple[bool, str]:
    """
    Check if OpenCL is available and working.

    Returns:
        Tuple[bool, str]: (availability status, error message if not available)
    """
    if not OPENCL_AVAILABLE:
        return False, f"PyOpenCL initialization failed: {init_error}"

    # Try to get platforms to check if OpenCL is working
    try:
        platforms = cl.get_platforms()
        if not platforms:
            return False, "No OpenCL platforms found. Check if drivers are installed."
        return True, ""
    except cl._cl.LogicError as e:
        if "PLATFORM_NOT_FOUND_KHR" in str(e):
            return False, "No OpenCL platforms found. You need to install appropriate OpenCL drivers for your hardware."
        return False, f"OpenCL error: {e!s}"
    except Exception as e:
        return False, f"Unexpected OpenCL error: {e!s}"


def list_devices() -> None:
    """List all available OpenCL platforms and devices."""
    available, error_msg = check_opencl_available()
    if not available:
        logger.error(f"Cannot list OpenCL devices: {error_msg}")
        logger.error("For NVIDIA GPUs: Install the NVIDIA GPU drivers and CUDA toolkit")
        logger.error("For AMD GPUs: Install the AMD drivers with OpenCL support")
        logger.error("For Intel GPUs: Install the Intel OpenCL runtime")
        return

    try:
        platforms = cl.get_platforms()
        if not platforms:
            logger.warning("No OpenCL platforms found!")
            return

        logger.info(f"Found {len(platforms)} OpenCL platform(s):")
        for i, platform in enumerate(platforms):
            logger.info(f"Platform {i}: {platform.name}")
            try:
                devices = platform.get_devices()
                for j, device in enumerate(devices):
                    logger.info(f"  Device {j}: {device.name} ({device.type})")
                    logger.info(f"    Compute Units: {device.max_compute_units}")
                    logger.info(f"    Global Memory: {device.global_mem_size / (1024**2):.2f} MB")
                    logger.info(f"    Local Memory: {device.local_mem_size / 1024:.2f} KB")
                    logger.info(f"    Max Work Group Size: {device.max_work_group_size}")
            except Exception as e:
                logger.error(f"  Error getting devices for platform {platform.name}: {e}")
    except Exception as e:
        logger.error(f"Error listing OpenCL devices: {e}")


def get_opencl_context(platform_index: int = 0, device_index: int = 0) -> tuple[Any, Any, Any]:
    """
    Initialize an OpenCL context.

    Args:
        platform_index (int): Index of the OpenCL platform to use
        device_index (int): Index of the OpenCL device to use

    Returns:
        Tuple containing the OpenCL context, device, and command queue

    Raises:
        RuntimeError: If OpenCL is not available or if the requested platform/device is not found
    """
    available, error_msg = check_opencl_available()
    if not available:
        raise RuntimeError(f"OpenCL is not available: {error_msg}")

    try:
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found!")

        if platform_index >= len(platforms):
            raise IndexError(f"Platform index {platform_index} out of range, only {len(platforms)} platforms available")

        platform = platforms[platform_index]
        devices = platform.get_devices()

        if not devices:
            raise RuntimeError(f"No OpenCL devices found on platform {platform.name}!")

        if device_index >= len(devices):
            raise IndexError(f"Device index {device_index} out of range, only {len(devices)} devices available")

        device = devices[device_index]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        logger.info(f"Using OpenCL device: {device.name} on platform: {platform.name}")
        return context, device, queue
    except cl._cl.LogicError as e:
        if "PLATFORM_NOT_FOUND_KHR" in str(e):
            raise RuntimeError("No OpenCL platforms found. You need to install appropriate OpenCL drivers for your hardware.")
        raise RuntimeError(f"OpenCL error: {e!s}")
    except Exception as e:
        raise RuntimeError(f"Error initializing OpenCL: {e!s}")


class CLProgram:

    """Class to manage an OpenCL program."""

    def __init__(self, context: Any, source_code: str) -> None:
        """
        Initialize the OpenCL program.

        Args:
            context (cl.Context): OpenCL context
            source_code (str): OpenCL kernel source code
        """
        self.context = context
        self.source_code = source_code
        self.program = cl.Program(context, source_code).build()

    def get_kernel(self, kernel_name: str) -> Any:
        """
        Get a kernel from the program.

        Args:
            kernel_name (str): Name of the kernel function

        Returns:
            cl.Kernel: OpenCL kernel
        """
        return getattr(self.program, kernel_name)


# OpenCL Kernel for particle simulation
PARTICLE_KERNEL = """
__kernel void apply_forces(
    __global float4 *positions_velocities,  // x, y, vx, vy
    __global int *colors,
    __global float *rules_array,
    __global float *radii2_array,
    const int num_atoms,
    const int num_colors,
    const float width,
    const float height,
    const float time_scale,
    const float viscosity,
    const float wall_repel,
    const float gravity,
    const int pulse,
    const float pulse_x,
    const float pulse_y,
    __global float *total_velocities
) {
    const int i = get_global_id(0);
    if (i >= num_atoms) return;

    // Extract data for this atom
    float4 atom = positions_velocities[i];
    float x = atom.x;
    float y = atom.y;
    float vx = atom.z;
    float vy = atom.w;
    int color = colors[i];

    // Calculate forces
    float fx = 0.0f;
    float fy = 0.0f;

    // Get color radius
    float color_radius2 = radii2_array[color];
    // Base index for rules
    int rule_base_idx = color * num_colors;

    // Calculate interactions with all other atoms
    for (int j = 0; j < num_atoms; j++) {
        if (i == j) continue;  // Skip self-interaction

        float4 other = positions_velocities[j];
        float dx = x - other.x;
        float dy = y - other.y;
        float d2 = dx*dx + dy*dy;

        // Check if in range
        if (d2 > 0 && d2 < color_radius2) {
            float d = sqrt(d2);
            int other_color = colors[j];
            float rule = rules_array[rule_base_idx + other_color];

            // Calculate force
            float force = rule / d;
            fx += (dx / d) * force;
            fy += (dy / d) * force;
        }
    }

    // Apply pulse if active
    if (pulse != 0) {
        float dx = x - pulse_x;
        float dy = y - pulse_y;
        float d2 = dx*dx + dy*dy;
        if (d2 > 0) {
            float pulse_force = 100.0f * pulse / (d2 * time_scale);
            fx += pulse_force * dx;
            fy += pulse_force * dy;
        }
    }

    // Apply wall repulsion
    if (wall_repel > 0) {
        if (x < wall_repel) {
            fx += (wall_repel - x) * 0.1f;
        }
        if (x > width - wall_repel) {
            fx += (width - wall_repel - x) * 0.1f;
        }
        if (y < wall_repel) {
            fy += (wall_repel - y) * 0.1f;
        }
        if (y > height - wall_repel) {
            fy += (height - wall_repel - y) * 0.1f;
        }
    }

    // Apply gravity
    fy += gravity;

    // Update velocity with forces and viscosity
    float vmix = 1.0f - viscosity;
    vx = vx * vmix + fx * time_scale;
    vy = vy * vmix + fy * time_scale;

    // Update position
    x += vx;
    y += vy;

    // Boundary conditions - bounce off walls
    if (x < 0) {
        x = -x;
        vx = -vx;
    }
    if (x >= width) {
        x = 2 * width - x;
        vx = -vx;
    }
    if (y < 0) {
        y = -y;
        vy = -vy;
    }
    if (y >= height) {
        y = 2 * height - y;
        vy = -vy;
    }

    // Store updated values
    positions_velocities[i] = (float4)(x, y, vx, vy);

    // Store absolute velocity for time scaling calculation
    total_velocities[i] = fabs(vx) + fabs(vy);
}
"""
