from color_constants import (MATERIAL_BLUE, MATERIAL_BLUE_STROKE,
                             MATERIAL_DARK_GRAY, MATERIAL_DARK_GRAY_STROKE,
                             MATERIAL_MINT, MATERIAL_MINT_STROKE, MATERIAL_RED,
                             MATERIAL_RED_STROKE, get_text_color,
                             get_token_dataset_color, get_token_model_color)
from manim import *
from monospace_text import MonospaceText
from round_box import RoundBox


class Tokens(VGroup):
    """Visualization of token vectors (e.g., 512 x 1, N x 768)."""

    def __init__(
        self,
        num_tokens=1,
        token_dim=512,
        box_width=0.8,
        box_height=2,
        spacing=0.3,
        colors=[MATERIAL_BLUE],
        stroke_colors=[MATERIAL_BLUE_STROKE],
        label=None,
        abbreviated=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.abbreviated = abbreviated

        # Create token boxes
        self.token_boxes = VGroup()
        for i in range(num_tokens):
            box = RoundBox(
                width=box_width,
                height=box_height,
                fill_color=colors[i % len(colors)],
                stroke_color=stroke_colors[i % len(stroke_colors)],
            )
            box.shift(RIGHT * i * (box_width + spacing))
            self.token_boxes.add(box)

        # Add "..." ellipsis if abbreviated
        if abbreviated:
            ellipsis_box = RoundBox(
                width=box_width,
                height=box_height,
                fill_color=MATERIAL_DARK_GRAY,
                stroke_color=MATERIAL_DARK_GRAY_STROKE,
                stroke_width=1,
            )
            ellipsis_text = MonospaceText("...", font_size=32, color=get_text_color())
            ellipsis_text.move_to(ellipsis_box.get_center())
            ellipsis_box.add(ellipsis_text)
            ellipsis_box.shift(RIGHT * num_tokens * (box_width + spacing))
            self.token_boxes.add(ellipsis_box)

        # Center the token boxes
        self.token_boxes.move_to(ORIGIN)

        self.add(self.token_boxes)

        # Add dimension label
        dim_text = f"{token_dim} × 1" if num_tokens == 1 else f"{num_tokens} x {token_dim} × 1"
        self.dim_label = MonospaceText(dim_text, font_size=20, color=get_text_color())
        self.dim_label.next_to(self.token_boxes, UP, buff=0.2)
        self.add(self.dim_label)

        # Add optional custom label
        if label:
            self.label = MonospaceText(label, font_size=24, color=get_text_color())
            self.label.next_to(self.token_boxes, DOWN, buff=0.3)
            self.add(self.label)


class ModelTokens(Tokens):
    """Specific visualization for model tokens."""

    def __init__(self, num_models=1, **kwargs):
        if 'colors' not in kwargs:
            kwargs['colors'] = [MATERIAL_RED]
        if 'stroke_colors' not in kwargs:
            kwargs['stroke_colors'] = [MATERIAL_RED_STROKE]

        super().__init__(
            num_tokens=num_models,
            token_dim=512,
            label="Model Tokens",
            **kwargs
        )


class DatasetTokens(Tokens):
    """Specific visualization for dataset tokens."""

    def __init__(self, num_samples=1, **kwargs):
        if 'colors' not in kwargs:
            kwargs['colors'] = [MATERIAL_MINT]
        if 'stroke_colors' not in kwargs:
            kwargs['stroke_colors'] = [MATERIAL_MINT_STROKE]

        super().__init__(
            num_tokens=num_samples,
            token_dim=512,
            label="Dataset Tokens",
            **kwargs
        )
