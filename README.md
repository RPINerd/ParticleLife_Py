# Particle Life Simulation in Python

A Python implementation of the Particle Life simulation using Pygame-CE and NumPy for efficient computation.

## Description

Based on [Particle Life](https://github.com/hunar4321/particle-life) by Hunar4321, this is an attempt to create an optimized and full-featured python version of the simulation.

Particle Life is a continuous cellular automata-like system where virtual particles interact with each other based on simple rules. The resulting patterns can show emergent behaviors that resemble simple life-like systems.

## Features

- ~~Efficient particle physics simulation using NumPy for vectorized operations~~
- Interactive UI with controls for all simulation parameters
- ~~Support for up to 7 different particle types with customizable interaction rules~~
- ~~Cluster detection and visualization~~

## Requirements

- Built on Python 3.13, but should work with 3.9 or higher
- Dependencies in requirements.txt:
  - pygame-ce (Pygame Community Edition)
  - numpy
  - pygame-gui
  - imageio (for video recording)
  - imageio-ffmpeg (for video encoding)

## Installation

1. Clone or download this repository
2. Create a virtual environment using UV (recommended):

   ```bash
   pip install uv  # Install UV if you don't have it yet
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies using UV:

   ```bash
   uv pip install -r requirements.txt
   ```

Alternatively, you can use traditional tools:

```bash
# With pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# With conda
conda create -n particlelife python=3.10
conda activate particlelife
pip install -r requirements.txt
```

## Usage

Run the simulation with default settings:

```bash
python main.py
```

### Command Line Options

- ~~`--seed SEED`: Set the random seed for consistent results~~
- `--width WIDTH`: Set window width in pixels
- `--height HEIGHT`: Set window height in pixels
- `--fullscreen`: Run in fullscreen mode
- `--colors COLORS`: Set number of particle colors (1-7)
- `--atoms-per-color COUNT`: Set number of atoms per color
- `--debug`: Enable debug logging

Example:

```bash
python main.py --seed 12345 --width 1200 --height 900 --colors 5
```

### Controls

- ~~**Mouse Click**: Create an attractive pulse at the mouse position~~
- ~~**Shift + Mouse Click**: Create a repulsive pulse at the mouse position~~
- **R Key**: Generate new random rules
- ~~**T Key**: Toggle cluster tracking~~
- **O Key**: Reset atom positions
- **S Key**: Make rules symmetric
- ~~**F Key**: Take a screenshot~~
- ~~**V Key**: Toggle video recording~~
- **ESC Key**: Quit the simulation

## How It Works

Each particle has a position, velocity, and color. The simulation applies forces between particles based on:

1. Their distance from each other
2. The interaction rule between their colors
3. The radius of interaction for each color

Positive rule values create attraction, while negative values create repulsion.
