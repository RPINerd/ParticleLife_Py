"""
UI module for Particle Life simulation.

Handles user interface elements and controls.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame
import pygame_gui
from pygame_gui.core import ObjectID

if TYPE_CHECKING:
    from particlelife.settings import Settings
logger = logging.getLogger(__name__)


class UI:

    """User interface for the Particle Life simulation."""

    def __init__(self, settings: "Settings", settings_changed_callback: Callable | None = None) -> None:
        """
        Initialize the UI.

        Args:
            settings (Settings): Simulation settings
            settings_changed_callback (callable): Function to call when settings are changed
        """
        self.settings = settings
        self.settings_changed_callback = settings_changed_callback

        # Initialize pygame_gui manager
        self.ui_manager = pygame_gui.UIManager(
            (settings.width, settings.height),
            theme_path="particlelife/data/theme.json"
        )

        # Create UI elements
        self._create_ui_elements()

    def _create_ui_elements(self) -> None:
        """Create the UI elements."""
        # Main panel
        panel_width = 250
        panel_height = self.settings.height - 20

        self.panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(
                (self.settings.width - panel_width - 10, 10),
                (panel_width, panel_height)
            ),
            manager=self.ui_manager,
            starting_height=1
        )

        # Title
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (panel_width - 20, 30)),
            text=f"Particle Life #{self.settings.seed}",
            manager=self.ui_manager,
            container=self.panel
        )

        # FPS Counter
        self.fps_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 50), (panel_width - 20, 30)),
            text=f"FPS: {self.settings.fps}",
            manager=self.ui_manager,
            container=self.panel
        )

        # Configuration controls
        y_pos = 90
        y_increment = 35

        # Buttons section
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Configuration",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Reset button
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (110, 30)),
            text="Reset",
            manager=self.ui_manager,
            container=self.panel,
            object_id=ObjectID("#reset_button")
        )

        # Random Rules button
        self.random_rules_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((130, y_pos), (110, 30)),
            text="Random Rules",
            manager=self.ui_manager,
            container=self.panel,
            object_id=ObjectID("#random_rules_button")
        )
        y_pos += y_increment

        # Symmetric Rules button
        self.symmetric_rules_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Symmetric Rules",
            manager=self.ui_manager,
            container=self.panel,
            object_id=ObjectID("#symmetric_rules_button")
        )
        y_pos += y_increment

        # Number of Colors slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Number of Colors",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.colors_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            start_value=self.settings.num_colors,
            value_range=(1, 7),
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Seed input
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Seed",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.seed_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            manager=self.ui_manager,
            container=self.panel,
            initial_text=str(self.settings.seed)
        )
        y_pos += y_increment

        # Atoms per color slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Atoms per Color",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.atoms_count_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            start_value=self.settings.atoms_settings["count"],
            value_range=(1, 1000),
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Time scale slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Time Scale",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.time_scale_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            start_value=self.settings.time_scale,
            value_range=(0.1, 5.0),
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Viscosity slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Viscosity",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.viscosity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            start_value=self.settings.viscosity,
            value_range=(0.1, 2.0),
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Gravity slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Gravity",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += 30

        self.gravity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            start_value=self.settings.gravity,
            value_range=(0.0, 1.0),
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Drawing options
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Drawing Options",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Circle shape checkbox
        self.circle_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Circle Shape: " + ("ON" if self.settings.drawings["circle"] else "OFF"),
            manager=self.ui_manager,
            container=self.panel,
            tool_tip_text="Draw particles as circles instead of squares",
            object_id=ObjectID("#toggle_button")
        )
        y_pos += y_increment

        # Lines checkbox
        self.lines_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Draw Lines: " + ("ON" if self.settings.drawings["lines"] else "OFF"),
            manager=self.ui_manager,
            container=self.panel,
            tool_tip_text="Draw lines between interacting particles",
            object_id=ObjectID("#toggle_button")
        )
        y_pos += y_increment

        # Clusters checkbox
        # self.clusters_checkbox = pygame_gui.elements.UIButton(
        #     relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
        #     text="Track Clusters: " + ("ON" if self.settings.drawings["clusters"] else "OFF"),
        #     manager=self.ui_manager,
        #     container=self.panel,
        #     tool_tip_text="Track and visualize particle clusters",
        #     object_id=ObjectID("#toggle_button")
        # )
        # y_pos += y_increment

        # Export options
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (panel_width - 20, 30)),
            text="Export",
            manager=self.ui_manager,
            container=self.panel
        )
        y_pos += y_increment

        # Screenshot button
        self.screenshot_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (110, 30)),
            text="Screenshot",
            manager=self.ui_manager,
            container=self.panel,
            object_id=ObjectID("#screenshot_button")
        )

        # Record video button
        self.record_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((130, y_pos), (110, 30)),
            text="Record Video",
            manager=self.ui_manager,
            container=self.panel,
            object_id=ObjectID("#record_button")
        )

    def process_event(self, event: pygame.event.Event) -> bool:
        """
        Process UI events.

        Args:
            event (pygame.event.Event): Event to process

        Returns:
            bool: True if event was processed, False otherwise
        """
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                self._handle_button_event(event)
                return True
            if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                self._handle_slider_event(event)
                return True
            if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                self._handle_text_entry_event(event)
                return True

        # Let pygame_gui handle the event if needed
        return self.ui_manager.process_events(event)

    def _handle_button_event(self, event: pygame.event.Event) -> None:
        """
        Handle button click events.

        Args:
            event (pygame.event.Event): Button event
        """
        if event.ui_element == self.reset_button:
            logger.info("Reset button clicked")
            if self.settings_changed_callback:
                self.settings_changed_callback()

        elif event.ui_element == self.random_rules_button:
            logger.info("Random rules button clicked")
            if self.settings_changed_callback:
                self.settings_changed_callback()

        elif event.ui_element == self.symmetric_rules_button:
            logger.info("Symmetric rules button clicked")
            if self.settings_changed_callback:
                self.settings_changed_callback()

        elif event.ui_element == self.circle_checkbox:
            self.settings.drawings["circle"] = not self.settings.drawings["circle"]
            self.circle_checkbox.set_text(
                "Circle Shape: " + ("ON" if self.settings.drawings["circle"] else "OFF")
            )

        elif event.ui_element == self.lines_checkbox:
            self.settings.drawings["lines"] = not self.settings.drawings["lines"]
            self.lines_checkbox.set_text(
                "Draw Lines: " + ("ON" if self.settings.drawings["lines"] else "OFF")
            )

        # elif event.ui_element == self.clusters_checkbox:
        #     self.settings.drawings["clusters"] = not self.settings.drawings["clusters"]
        #     self.clusters_checkbox.set_text(
        #         "Track Clusters: " + ("ON" if self.settings.drawings["clusters"] else "OFF")
        #     )

        elif event.ui_element == self.screenshot_button:
            logger.info("Screenshot button clicked")
            # Handled by simulation

        elif event.ui_element == self.record_button:
            logger.info("Record button clicked")
            # Handled by simulation

    def _handle_slider_event(self, event: pygame.event.Event) -> None:
        """
        Handle slider movement events.

        Args:
            event (pygame.event.Event): Slider event
        """
        if event.ui_element == self.colors_slider:
            value = int(event.value)
            if value != self.settings.num_colors:
                self.settings.num_colors = value
                logger.info(f"Number of colors changed to {value}")
                if self.settings_changed_callback:
                    self.settings_changed_callback()

        elif event.ui_element == self.atoms_count_slider:
            value = int(event.value)
            if value != self.settings.atoms_settings["count"]:
                self.settings.atoms_settings["count"] = value
                logger.info(f"Atoms per color changed to {value}")
                if self.settings_changed_callback:
                    self.settings_changed_callback()

        elif event.ui_element == self.time_scale_slider:
            self.settings.time_scale = float(event.value)

        elif event.ui_element == self.viscosity_slider:
            self.settings.viscosity = float(event.value)

        elif event.ui_element == self.gravity_slider:
            self.settings.gravity = float(event.value)

    def _handle_text_entry_event(self, event: pygame.event.Event) -> None:
        """
        Handle text entry events.

        Args:
            event (pygame.event.Event): Text entry event
        """
        if event.ui_element == self.seed_entry:
            try:
                value = int(event.text)
                if value != self.settings.seed:
                    self.settings.seed = value
                    logger.info(f"Seed changed to {value}")
                    if self.settings_changed_callback:
                        self.settings_changed_callback()
            except ValueError:
                # Reset to current seed if invalid input
                self.seed_entry.set_text(str(self.settings.seed))

    def draw(self) -> None:
        """Draw the UI elements."""
        # Update FPS display
        self.fps_label.set_text(f"FPS: {self.settings.fps}")

        # Update UI
        self.ui_manager.update(1 / 60.0)  # Fixed time step for UI

        # Draw UI
        self.ui_manager.draw_ui(pygame.display.get_surface())

    def rebuild_ui(self) -> None:
        """
        Rebuild the UI (for example after changing colors or rules).

        Usually called after significant settings changes.
        """
        self.ui_manager.clear_and_reset()
        self._create_ui_elements()
