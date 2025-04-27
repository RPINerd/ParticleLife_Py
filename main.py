"""
Particle Life Simulation | RPINerd, 04/2025

A python implementation of the Particle Life simulation made by Hunar4321. With a focus on feature pairity
with the original JS/CPP version, while having minimal compromise on performance.
"""

import argparse
import logging
import sys

from particlelife.opencl_utils import check_opencl_available, list_devices
from particlelife.simulation import Simulation


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Particle Life Simulation")
    parser.add_argument(
        "--seed", type=int, default=91651088029, help="Random seed for simulation"
    )
    parser.add_argument(
        "--width", type=int, default=800, help="Width of the simulation window"
    )
    parser.add_argument(
        "--height", type=int, default=600, help="Height of the simulation window"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--colors", type=int, default=4, help="Number of particle colors", choices=range(1, 8)
    )
    parser.add_argument(
        "--atoms-per-color", type=int, default=500, help="Number of atoms per color"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    # OpenCL arguments
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration via OpenCL"
    )
    parser.add_argument(
        "--platform-index", type=int, default=0,
        help="OpenCL platform index to use (default: 0)"
    )
    parser.add_argument(
        "--device-index", type=int, default=0,
        help="OpenCL device index to use (default: 0)"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available OpenCL platforms and devices and exit"
    )
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        debug (bool): Whether to enable debug logging
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    setup_logging(args.debug)

    # List devices if requested
    if args.list_devices:
        list_devices()
        return

    # Check OpenCL availability if GPU mode requested
    if args.use_gpu:
        available, error_msg = check_opencl_available()
        if not available:
            logging.error(f"OpenCL not available: {error_msg}")
            logging.error("Falling back to CPU mode")
            logging.error("To use GPU acceleration, install appropriate OpenCL drivers for your hardware:")
            logging.error("- For NVIDIA GPUs: Install NVIDIA drivers and CUDA toolkit")
            logging.error("- For AMD GPUs: Install AMD drivers with OpenCL support")
            logging.error("- For Intel GPUs: Install Intel OpenCL runtime")
            args.use_gpu = False

    sim = Simulation(
        seed=args.seed,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        num_colors=args.colors,
        atoms_per_color=args.atoms_per_color,
        use_gpu=args.use_gpu,
        platform_index=args.platform_index,
        device_index=args.device_index,
    )

    sys.exit(sim.run())


if __name__ == "__main__":
    main()
