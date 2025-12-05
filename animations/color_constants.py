"""Color constants for animations with light/dark theme support."""
from manim import *

# Theme mode: "light" or "dark"
THEME_MODE = "light"


class ColorTheme:
    """Color theme manager for light and dark modes."""

    # Material Design Colors
    MATERIAL_WHITE = "#FFFFFF"
    MATERIAL_WHITE_STROKE = "#C3CFD9"

    MATERIAL_GRAY = "#F2F5F7"
    MATERIAL_GRAY_STROKE = "#C3CFD9"

    MATERIAL_DARK_GRAY = "#E1E5EC"
    MATERIAL_DARK_GRAY_STROKE = "#9EADBA"

    MATERIAL_GREEN = "#D2E4E1"
    MATERIAL_GREEN_STROKE = "#207868"

    MATERIAL_MINT = "#D1EFEC"
    MATERIAL_MINT_STROKE = "#1AAE9F"

    MATERIAL_BLUE = "#D5E7F7"
    MATERIAL_BLUE_STROKE = "#2C88D9"

    MATERIAL_PURPLE = "#F2D6F6"
    MATERIAL_PURPLE_STROKE = "#BD34D1"

    MATERIAL_ORANGE = "#FAE6D8"
    MATERIAL_ORANGE_STROKE = "#E8833A"

    MATERIAL_RED = "#F6DADE"
    MATERIAL_RED_STROKE = "#D3455B"

    MATERIAL_YELLOW = "#FDF3D3"
    MATERIAL_YELLOW_STROKE = "#F7C325"

    MATERIAL_PINK = "#F2D6F6"
    MATERIAL_PINK_STROKE = "#BD34D1"

    MATERIAL_INDIGO = "#E0DEFD"
    MATERIAL_INDIGO_STROKE = "#6558F5"

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
        "token_dataset": MATERIAL_MINT,

        "clustering": MATERIAL_GREEN,
        "sampling": MATERIAL_BLUE,
        "sample_box": MATERIAL_ORANGE,

        "encoder_image": MATERIAL_MINT,
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


# Stroke color getters
def get_green_stroke():
    """Get green stroke color."""
    return ColorTheme.MATERIAL_GREEN_STROKE


def get_mint_stroke():
    """Get mint/teal stroke color."""
    return ColorTheme.MATERIAL_MINT_STROKE


def get_blue_stroke():
    """Get blue stroke color."""
    return ColorTheme.MATERIAL_BLUE_STROKE


def get_purple_stroke():
    """Get purple stroke color."""
    return ColorTheme.MATERIAL_PURPLE_STROKE


def get_orange_stroke():
    """Get orange stroke color."""
    return ColorTheme.MATERIAL_ORANGE_STROKE


def get_red_stroke():
    """Get red stroke color."""
    return ColorTheme.MATERIAL_RED_STROKE


def get_yellow_stroke():
    """Get yellow stroke color."""
    return ColorTheme.MATERIAL_YELLOW_STROKE


def get_pink_stroke():
    """Get pink stroke color."""
    return ColorTheme.MATERIAL_PINK_STROKE


def get_indigo_stroke():
    """Get indigo stroke color."""
    return ColorTheme.MATERIAL_INDIGO_STROKE


# Direct color access (for backward compatibility)
MATERIAL_WHITE = ColorTheme.MATERIAL_WHITE
MATERIAL_WHITE_STROKE = ColorTheme.MATERIAL_WHITE_STROKE

MATERIAL_GRAY = ColorTheme.MATERIAL_GRAY
MATERIAL_GRAY_STROKE = ColorTheme.MATERIAL_GRAY_STROKE

MATERIAL_DARK_GRAY = ColorTheme.MATERIAL_DARK_GRAY
MATERIAL_DARK_GRAY_STROKE = ColorTheme.MATERIAL_DARK_GRAY_STROKE

MATERIAL_GREEN = ColorTheme.MATERIAL_GREEN
MATERIAL_GREEN_STROKE = ColorTheme.MATERIAL_GREEN_STROKE

MATERIAL_MINT = ColorTheme.MATERIAL_MINT
MATERIAL_MINT_STROKE = ColorTheme.MATERIAL_MINT_STROKE

MATERIAL_BLUE = ColorTheme.MATERIAL_BLUE
MATERIAL_BLUE_STROKE = ColorTheme.MATERIAL_BLUE_STROKE

MATERIAL_PURPLE = ColorTheme.MATERIAL_PURPLE
MATERIAL_PURPLE_STROKE = ColorTheme.MATERIAL_PURPLE_STROKE

MATERIAL_ORANGE = ColorTheme.MATERIAL_ORANGE
MATERIAL_ORANGE_STROKE = ColorTheme.MATERIAL_ORANGE_STROKE

MATERIAL_RED = ColorTheme.MATERIAL_RED
MATERIAL_RED_STROKE = ColorTheme.MATERIAL_RED_STROKE

MATERIAL_YELLOW = ColorTheme.MATERIAL_YELLOW
MATERIAL_YELLOW_STROKE = ColorTheme.MATERIAL_YELLOW_STROKE

MATERIAL_PINK = ColorTheme.MATERIAL_PINK
MATERIAL_PINK_STROKE = ColorTheme.MATERIAL_PINK_STROKE

MATERIAL_INDIGO = ColorTheme.MATERIAL_INDIGO
MATERIAL_INDIGO_STROKE = ColorTheme.MATERIAL_INDIGO_STROKE
