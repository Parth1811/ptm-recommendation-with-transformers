from color_constants import *
from manim import *
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox


class AttentionBlock(VGroup):
    """Cross-attention mechanism visualization with Q, K, V."""

    def __init__(
        self,
        width=3,
        height=3,
        mode="cross",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Main attention block
        if mode == "cross":
            content = "Cross\nAttention\nBlock"
            fill_color = MATERIAL_GREEN
            stroke_color = MATERIAL_GREEN_STROKE
        else:
            content = "Self\nAttention\nBlock"
            fill_color = MATERIAL_RED
            stroke_color = MATERIAL_RED_STROKE

        self.block = RoundBox(
            content=content,
            width=width,
            height=height,
            fill_color=fill_color,
            stroke_color=stroke_color,
            fill_with_stroke=True,
            fill_opacity=0.6,
            text_align="center",
            font_size=32,
        )
        self.add(self.block)

        # Q, K, V input labels
        self.q_label = MonospaceText("Q", font_size=28, color=MATERIAL_YELLOW_STROKE)
        self.k_label = MonospaceText("K", font_size=28, color=MATERIAL_BLUE_STROKE)
        self.v_label = MonospaceText("V", font_size=28, color=MATERIAL_RED_STROKE)

        # Position inputs on the left side
        self.q_label.next_to(self.block, LEFT, buff=0.5).shift(UP * 0.8)
        self.k_label.next_to(self.block, LEFT, buff=0.5)
        self.v_label.next_to(self.block, LEFT, buff=0.5).shift(DOWN * 0.8)

        self.add(self.q_label, self.k_label, self.v_label)

        # Input arrows
        self.q_arrow = Line(self.q_label.get_right(), self.block.get_left() + UP * 0.8, buff=0.1, color=MATERIAL_YELLOW_STROKE)
        self.k_arrow = Line(self.k_label.get_right(), self.block.get_left(), buff=0.1, color=MATERIAL_BLUE_STROKE)
        self.v_arrow = Line(self.v_label.get_right(), self.block.get_left() + DOWN * 0.8, buff=0.1, color=MATERIAL_RED_STROKE)

        self.add(self.q_arrow, self.k_arrow, self.v_arrow)

        # Output label and arrow
        # self.output_label = Text("Output", font_size=24)
        # self.output_label.next_to(self.block, RIGHT, buff=0.5)
        # self.output_arrow = Arrow(self.block.get_right(), self.output_label.get_left(), buff=0.1)

        # self.add(self.output_label, self.output_arrow)


class Transformer(VGroup):
    """Transformer architecture with cross-attention and FC layer."""

    def __init__(
        self,
        show_fc_layer=True,
        mode="cross",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Cross-attention block
        self.attention = AttentionBlock(mode=mode)
        self.add(self.attention)

        if show_fc_layer:
            # Fully Connected Layer as Neural Network
            self.fc_layer = NeuralNetwork(
                layers=[4, 2, 1],
                layer_spacing=1.5,
                node_radius=0.15,
                node_opacity=0.8,
                show_labels=False,
            )
            self.fc_layer.scale(0.8)
            self.fc_layer.next_to(self.attention, RIGHT, buff=1.5)

            # Arrow from attention to FC
            self.attention_to_fc_arrow = Arrow(
                self.attention.get_right() + RIGHT * 0.2,
                self.fc_layer.get_left() + LEFT * 0.2,
                color=get_arrow_color(),
                buff=0.1
            )

            self.add(self.fc_layer, self.attention_to_fc_arrow)

            # FC Layer label
            self.fc_label = MonospaceText("Fully Connected", font_size=20, color=get_text_color())
            self.fc_label.next_to(self.fc_layer, DOWN, buff=0.3)
            self.add(self.fc_label)

            # Output dimensions
            self.output_dim = MonospaceText("N × 1", font_size=24, color=get_text_color())
            self.output_dim.next_to(self.fc_layer, UP, buff=0.5)
            self.add(self.output_dim)

        # Add positional encoding label
        # self.pos_encoding = Text("Positional Encoding", font_size=18, color=GRAY)
        # self.pos_encoding.next_to(self.attention, DOWN, buff=0.5)
        # self.add(self.pos_encoding)

        # Center the entire transformer
        self.move_to(ORIGIN)
