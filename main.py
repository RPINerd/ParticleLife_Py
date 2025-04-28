"""
Particle Life Simulation | RPINerd, 04/2025

A python implementation of the Particle Life simulation made by Hunar4321. With a focus on feature pairity
with the original JS/CPP version, while having minimal compromise on performance.
"""

import argparse
import logging
import sys

import pygame

from particlelife.settings import Settings
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


def main(args: argparse.Namespace) -> None:
    """Main entry point for the application."""
    settings = Settings(
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_colors=args.colors,
        atoms_per_color=args.atoms_per_color
        )

    # Setup display
    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    screen = pygame.display.set_mode(
        (args.width, args.height),
        flags
    )
    pygame.display.set_caption(f"Particle Life #{settings.seed}")

    sim = Simulation(
        settings=settings,
        screen=screen
    )

    sys.exit(sim.run())


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.debug)
    main(args)
