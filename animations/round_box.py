from manim import *


class RoundBox(VGroup):
    """Rounded rectangle container for text or other mobjects."""

    def __init__(
        self,
        content=None,
        width=2,
        height=1,
        corner_radius=0.1,
        fill_color=BLUE,
        fill_opacity=0.3,
        stroke_color=WHITE,
        stroke_width=2,
        **kwargs
    ):
        super().__init__(**kwargs)

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
                self.content = Text(content, font_size=24)
            else:
                self.content = content

            self.content.move_to(self.box.get_center())
            self.add(self.content)
