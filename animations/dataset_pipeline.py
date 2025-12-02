from manim import *
from round_box import RoundBox


class DatasetPipeline(VGroup):
    """Complete dataset processing pipeline visualization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unstructured dataset
        self.unstructured = self._create_data_grid("Unstructured\nDataset", 3, 3)
        self.add(self.unstructured)

        # Clustering box
        self.clustering1 = self._create_vertical_box("C\nL\nU\nS\nT\nE\nR\nI\nN\nG")
        self.clustering1.next_to(self.unstructured, RIGHT, buff=0.3)
        self.add(self.clustering1)

        # Arrow
        arrow1 = Arrow(
            self.unstructured.get_right(),
            self.clustering1.get_left(),
            buff=0.1,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        )
        self.add(arrow1)

        # Structured dataset
        self.structured = self._create_data_grid("Structured\nDataset", 3, 2)
        self.structured.next_to(self.clustering1, RIGHT, buff=0.5)
        self.add(self.structured)

        # Arrow
        arrow2 = Arrow(
            self.clustering1.get_right(),
            self.structured.get_left(),
            buff=0.1,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        )
        self.add(arrow2)

        # Sampling box
        self.sampling = self._create_vertical_box("S\nA\nM\nP\nL\nI\nN\nG")
        self.sampling.next_to(self.structured, RIGHT, buff=0.3)
        self.add(self.sampling)

        # Arrow
        arrow3 = Arrow(
            self.structured.get_right(),
            self.sampling.get_left(),
            buff=0.1,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        )
        self.add(arrow3)

        # Representative samples
        self.samples = VGroup()
        sample_labels = ["Q1", "Q2", "Q3"]
        for i, label in enumerate(sample_labels):
            sample = RoundBox(
                content=label,
                width=0.6,
                height=0.6,
                fill_color=ORANGE,
                fill_opacity=0.5,
                stroke_width=3,
            )
            sample.shift(DOWN * i * 0.8)
            self.samples.add(sample)

        self.samples.next_to(self.sampling, RIGHT, buff=0.5)
        self.add(self.samples)

        # Arrow
        arrow4 = Arrow(
            self.sampling.get_right(),
            self.samples.get_left(),
            buff=0.1,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        )
        self.add(arrow4)

        # Encoders
        self.encoders = VGroup()

        self.image_encoder = RoundBox(
            content="Image\nEncoder",
            width=1.5,
            height=1.2,
            fill_color=TEAL,
            fill_opacity=0.5,
            stroke_width=3,
        )

        self.text_encoder = RoundBox(
            content="Text\nEncoder",
            width=1.5,
            height=1.2,
            fill_color=PURPLE,
            fill_opacity=0.5,
            stroke_width=3,
        )

        self.image_encoder.next_to(self.samples, RIGHT, buff=1)
        self.text_encoder.next_to(self.image_encoder, DOWN, buff=0.5)

        self.encoders.add(self.image_encoder, self.text_encoder)
        self.add(self.encoders)

        # Label
        self.samples_label = Text("Representative\nSamples", font_size=18)
        self.samples_label.next_to(self.samples, DOWN, buff=0.3)
        self.add(self.samples_label)

        # Dataset tokens output
        self.tokens_label = Text("Dataset Tokens", font_size=20)
        self.tokens_label.next_to(self.encoders, RIGHT, buff=1)
        self.add(self.tokens_label)

        # Center the entire pipeline
        self.move_to(ORIGIN)

    def _create_data_grid(self, label, rows, cols):
        """Create a grid of small squares representing data."""
        group = VGroup()

        grid = VGroup()
        square_size = 0.3
        for i in range(rows):
            for j in range(cols):
                square = Square(side_length=square_size, stroke_width=3, stroke_color=WHITE)
                square.shift(RIGHT * j * (square_size + 0.05) + DOWN * i * (square_size + 0.05))
                grid.add(square)

        label_text = Text(label, font_size=18)
        label_text.next_to(grid, UP, buff=0.2)

        group.add(grid, label_text)
        return group

    def _create_vertical_box(self, text):
        """Create a vertical colored box with text."""
        return RoundBox(
            content=text,
            width=0.8,
            height=2.5,
            fill_color=GREEN,
            fill_opacity=0.5,
            stroke_width=3,
        )
