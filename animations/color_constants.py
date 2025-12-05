"""Color constants for animations with light/dark theme support."""
from manim import *

# Theme mode: "light" or "dark"
THEME_MODE = "light"


class ColorTheme:
    """Color theme manager for light and dark modes."""

    # Material Design Colors
    MATERIAL_GREEN = "#91DE93"
    MATERIAL_TEAL = "#009688"
    MATERIAL_BLUE = "#2196F3"
    MATERIAL_PURPLE = "#9C27B0"
    MATERIAL_ORANGE = "#FF9800"
    MATERIAL_RED = "#F44336"
    MATERIAL_YELLOW = "#FFEB3B"
    MATERIAL_PINK = "#E91E63"
    MATERIAL_CYAN = "#00BCD4"
    MATERIAL_LIME = "#CDDC39"
    MATERIAL_INDIGO = "#3F51B5"

    # Light Mode Colors
    LIGHT_THEME = {
        "background": "#FFFFFF",
        "text": "#000000",
        "text_secondary": "#666666",
        "stroke": "#333333",
        "grid": "#CCCCCC",
        "arrow": "#888888",

        # Component colors
        "neural_node": MATERIAL_BLUE,
        "neural_edge": "#2196F3",

        "token_model": MATERIAL_PINK,
        "token_dataset": MATERIAL_TEAL,

        "clustering": MATERIAL_GREEN,
        "sampling": MATERIAL_BLUE,
        "sample_box": MATERIAL_ORANGE,

        "encoder_image": MATERIAL_TEAL,
        "encoder_text": MATERIAL_PURPLE,

        "attention_block": "#4CAF50",  # LOGO_GREEN equivalent
        "fc_layer": MATERIAL_BLUE,

        "box_default": MATERIAL_BLUE,
    }

    # Dark Mode Colors
    DARK_THEME = {
        "background": "#000000",
        "text": "#FFFFFF",
        "text_secondary": "#AAAAAA",
        "stroke": "#FFFFFF",
        "grid": "#444444",
        "arrow": "#999999",

        # Component colors
        "neural_node": BLUE_A,  # Lighter blue for dark mode
        "neural_edge": "#42A5F5",

        "token_model": PURPLE_A,  # Lighter pink
        "token_dataset": TEAL_A,  # Lighter teal

        "clustering": "#A5D6A7",  # Lighter green
        "sampling": "#64B5F6",  # Lighter blue
        "sample_box": "#FFB74D",  # Lighter orange

        "encoder_image": "#4DB6AC",  # Lighter teal
        "encoder_text": "#BA68C8",  # Lighter purple

        "attention_block": GREEN_A,  # Lighter green
        "fc_layer": "#64B5F6",  # Lighter blue

        "box_default": "#64B5F6",  # Lighter blue
    }

    @classmethod
    def get_color(cls, color_name):
        """Get color based on current theme mode."""
        if THEME_MODE == "light":
            return cls.LIGHT_THEME.get(color_name, cls.MATERIAL_BLUE)
        else:
            return cls.DARK_THEME.get(color_name, cls.MATERIAL_BLUE)

    @classmethod
    def set_theme(cls, mode):
        """Set theme mode: 'light' or 'dark'."""
        global THEME_MODE
        THEME_MODE = mode


# Convenience functions for getting themed colors
def get_background_color():
    """Get background color for current theme."""
    return ColorTheme.get_color("background")


def get_text_color():
    """Get primary text color for current theme."""
    return ColorTheme.get_color("text")


def get_text_secondary_color():
    """Get secondary text color for current theme."""
    return ColorTheme.get_color("text_secondary")


def get_stroke_color():
    """Get stroke/border color for current theme."""
    return ColorTheme.get_color("stroke")


def get_grid_color():
    """Get grid color for current theme."""
    return ColorTheme.get_color("grid")


def get_arrow_color():
    """Get arrow color for current theme."""
    return ColorTheme.get_color("arrow")


# Neural Network Colors
def get_neural_node_color():
    """Get neural network node color."""
    return ColorTheme.get_color("neural_node")


def get_neural_edge_color():
    """Get neural network edge color."""
    return ColorTheme.get_color("neural_edge")


# Token Colors
def get_token_model_color():
    """Get model token color."""
    return ColorTheme.get_color("token_model")


def get_token_dataset_color():
    """Get dataset token color."""
    return ColorTheme.get_color("token_dataset")


# Pipeline Colors
def get_clustering_color():
    """Get clustering box color."""
    return ColorTheme.get_color("clustering")


def get_sampling_color():
    """Get sampling box color."""
    return ColorTheme.get_color("sampling")


def get_sample_box_color():
    """Get sample box color."""
    return ColorTheme.get_color("sample_box")


def get_encoder_image_color():
    """Get image encoder color."""
    return ColorTheme.get_color("encoder_image")


def get_encoder_text_color():
    """Get text encoder color."""
    return ColorTheme.get_color("encoder_text")


# Transformer Colors
def get_attention_block_color():
    """Get attention block color."""
    return ColorTheme.get_color("attention_block")


def get_fc_layer_color():
    """Get fully connected layer color."""
    return ColorTheme.get_color("fc_layer")


def get_box_default_color():
    """Get default box color."""
    return ColorTheme.get_color("box_default")


# Direct color access (for backward compatibility)
MATERIAL_GREEN = ColorTheme.MATERIAL_GREEN
MATERIAL_TEAL = ColorTheme.MATERIAL_TEAL
MATERIAL_BLUE = ColorTheme.MATERIAL_BLUE
MATERIAL_PURPLE = ColorTheme.MATERIAL_PURPLE
MATERIAL_ORANGE = ColorTheme.MATERIAL_ORANGE
MATERIAL_RED = ColorTheme.MATERIAL_RED
MATERIAL_YELLOW = ColorTheme.MATERIAL_YELLOW
MATERIAL_PINK = ColorTheme.MATERIAL_PINK
MATERIAL_CYAN = ColorTheme.MATERIAL_CYAN
MATERIAL_LIME = ColorTheme.MATERIAL_LIME
MATERIAL_INDIGO = ColorTheme.MATERIAL_INDIGO
