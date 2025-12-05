from color_constants import get_box_default_color, get_stroke_color
from manim import *
from monospace_text import MonospaceText


class RoundBox(VGroup):
    """Rounded rectangle container for text or other mobjects."""

    def __init__(
        self,
        content=None,
        width=2,
        height=1,
        corner_radius=0.1,
        fill_color=None,
        fill_opacity=0.8,
        stroke_color=None,
        stroke_width=2,
        text_align="center",
        font_size=24,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Set default colors if not provided
        if fill_color is None:
            fill_color = get_box_default_color()
        if stroke_color is None:
            stroke_color = get_stroke_color()

        self.box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=corner_radius,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

        self.add(self.box)

        if content is not None:
            if isinstance(content, str):
                # Map text_align to Manim's alignment parameter
                alignment_map = {
                    "left": LEFT,
                    "center": ORIGIN,
                    "right": RIGHT
                }
                align_point = alignment_map.get(text_align.lower(), ORIGIN)

                self.content = MonospaceText(content, font_size=font_size, color=get_stroke_color())
                # Align text within the box
                if text_align.lower() == "center":
                    self.content.move_to(self.box.get_center())
                elif text_align.lower() == "left":
                    self.content.move_to(self.box.get_center())
                    self.content.align_to(self.box, LEFT)
                    self.content.shift(RIGHT * 0.2)  # Small padding
                elif text_align.lower() == "right":
                    self.content.move_to(self.box.get_center())
                    self.content.align_to(self.box, RIGHT)
                    self.content.shift(LEFT * 0.2)  # Small padding
            else:
                self.content = content
                self.content.move_to(self.box.get_center())

            self.add(self.content)
