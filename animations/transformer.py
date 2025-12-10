import random

from manim import *

from color_constants import *
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import Tokens


class ProbabilityDistribution(VGroup):
    """Probability distribution visualization with stacked boxes and PDF curve."""

    def __init__(self, num_visible=4, total_items=10, attention_height=3.0, seed=42, **kwargs):
        super().__init__(**kwargs)

        self.num_visible = num_visible
        self.total_items = total_items

        random.seed(seed)

        # Generate random probabilities that sum to 1
        raw_probs = [random.random() for _ in range(total_items)]
        total = sum(raw_probs)
        self.probabilities = [p / total for p in raw_probs]

        # Box dimensions - height matches attention block, no spacing (connected boxes)
        box_width = 1.0
        total_height = attention_height
        # Calculate box height to fit num_visible + 1 (for ellipsis) boxes in total_height
        box_height = total_height / (num_visible + 1)

        # Create boxes for visible items (connected, no spacing)
        self.boxes = VGroup()

        # Show first few boxes
        for i in range(num_visible // 2):
            self._create_box(i, box_width, box_height)

        # Add ellipsis box
        ellipsis_box = RoundBox(
            content="...",
            width=box_width,
            height=box_height,
            fill_color=MATERIAL_DARK_GRAY,
            stroke_color=MATERIAL_DARK_GRAY_STROKE,
            stroke_width=2,
            font_size=20,
            corner_radius=0.0,  # No rounding for connected boxes
        )
        ellipsis_y = -(num_visible // 2) * box_height
        ellipsis_box.shift(UP * ellipsis_y)
        self.boxes.add(ellipsis_box)

        # Show last few boxes
        for i in range(num_visible // 2):
            idx = total_items - (num_visible // 2) + i
            offset = (num_visible // 2) + 1 + i
            self._create_box(idx, box_width, box_height, offset)

        self.add(self.boxes)

        # Create PDF curve on the right
        self.pdf_curve = self._create_pdf_curve(box_width, box_height, num_visible, total_items)
        if self.pdf_curve:
            self.add(self.pdf_curve)

        # Always add "Output PDF" label
        self.label = MonospaceText("Output PDF", font_size=20, color=get_text_color())
        self.label.next_to(self.boxes, DOWN, buff=0.3)
        self.add(self.label)

    def _create_box(self, idx, box_width, box_height, offset=None):
        """Create a box with probability value."""
        if offset is None:
            offset = idx

        prob = self.probabilities[idx]
        prob_text = f"{prob:.2f}"

        box = RoundBox(
            content=prob_text,
            width=box_width,
            height=box_height,
            fill_color=MATERIAL_BLUE,
            stroke_color=MATERIAL_BLUE_STROKE,
            stroke_width=2,
            font_size=18,
            corner_radius=0.0,  # No rounding for connected boxes
        )

        # Position box vertically (no spacing between boxes)
        y_pos = -offset * box_height
        box.shift(UP * y_pos)
        self.boxes.add(box)

    def _create_pdf_curve(self, box_width, box_height, num_visible, total_items):
        """Create smooth PDF curve on the right side in blue.

        Returns a VGroup containing the curve that can be positioned independently.
        """
        # Calculate positions for the curve (relative to origin)
        points = []

        # Add points for first visible boxes (at their center y position)
        for i in range(num_visible // 2):
            y_pos = -i * box_height - box_height / 2
            x_pos = self.probabilities[i] * 2.0 / max(self.probabilities)
            points.append([box_width / 2 + x_pos + 0.5, y_pos, 0])

        # Add point for middle (ellipsis)
        middle_y = -(num_visible // 2) * box_height - box_height / 2
        middle_idx = total_items // 2
        middle_x = self.probabilities[middle_idx] * 2.0 / max(self.probabilities)
        points.append([box_width / 2 + middle_x + 0.5, middle_y, 0])

        # Add points for last visible boxes
        for i in range(num_visible // 2):
            idx = total_items - (num_visible // 2) + i
            offset = (num_visible // 2) + 1 + i
            y_pos = -offset * box_height - box_height / 2
            x_pos = self.probabilities[idx] * 2.0 / max(self.probabilities)
            points.append([box_width / 2 + x_pos + 0.5, y_pos, 0])

        # Create smooth curve through points
        if len(points) >= 2:
            curve = VMobject()
            curve.set_points_smoothly([np.array(p) for p in points])
            curve.set_color(MATERIAL_BLUE_STROKE)  # Blue color
            curve.set_stroke(width=5)

            # Wrap in VGroup for positioning flexibility
            curve_group = VGroup(curve)
            return curve_group

        return None

    def animate_sort_descending(self, scene, duration=2.0):
        """Animate rearranging box content in descending probability order.

        Boxes stay in their positions, but content is swapped to show
        descending order from top to bottom.
        """
        # Sort probabilities descending
        sorted_probs = sorted(self.probabilities, reverse=True)

        box_transform_animations = []
        # Create transform animations for boxes
        for i in range(len(self.boxes)):
            if i == len(self.boxes) // 2:
                box_transform_animations.append(Indicate(self.boxes[i], color=MATERIAL_DARK_GRAY))
                continue
            old_box = self.boxes[len(self.boxes) - 1 - i]
            old_box_reverse = self.boxes[i]

            new_box = RoundBox(
                content=f"{sorted_probs[i]:.2f}",
                width=old_box.width,
                height=old_box.height,
                fill_color=MATERIAL_BLUE,
                stroke_color=MATERIAL_BLUE_STROKE,
                stroke_width=2,
                font_size=18,
                corner_radius=0.0,  # No rounding for connected boxes
            )
            new_box.move_to(old_box_reverse.get_center())
            new_box.set_opacity(1.0)

            box_transform_animations.append(
                Transform(old_box, new_box)
            )

        # Update probabilities to sorted order
        self.probabilities = sorted_probs

        # Create new PDF curve with sorted probabilities
        new_pdf_curve = self._create_pdf_curve(
            box_width=self.boxes[0].width,
            box_height=self.boxes[0].height,
            num_visible=self.num_visible,
            total_items=len(self.probabilities)
        )

        # Position new curve at same location as old curve
        if new_pdf_curve and self.pdf_curve:
            new_pdf_curve.move_to(self.pdf_curve.get_center())

        # Animate transformations in parallel
        curve_anim = Transform(self.pdf_curve, new_pdf_curve) if new_pdf_curve else []

        new_label = MonospaceText("Rankings", font_size=20, color=get_text_color())
        new_label.move_to(self.label.get_center())

        scene.play(
            *box_transform_animations,
            curve_anim,
            Transform(self.label, new_label),
            run_time=duration
        )

        rank_transform_animations = []
        for i in range(len(self.boxes)):
            if i == len(self.boxes) // 2:
                old_box = self.boxes[i]
                new_box = ellipsis_box = RoundBox(
                    content="...",
                    width=old_box.width,
                    height=old_box.height,
                    fill_color=MATERIAL_YELLOW,
                    stroke_color=MATERIAL_YELLOW_STROKE,
                    stroke_width=2,
                    font_size=20,
                    corner_radius=0.0,  # No rounding for connected boxes
                )
                new_box.move_to(old_box.get_center())
                rank_transform_animations.append(
                    Transform(old_box, new_box)
                )
                continue
            old_box = self.boxes[i]

            new_box = RoundBox(
                content=f"{len(self.boxes) - i}",
                width=old_box.width,
                height=old_box.height,
                fill_color=MATERIAL_YELLOW,
                stroke_color=MATERIAL_YELLOW_STROKE,
                stroke_width=2,
                font_size=18,
                corner_radius=0.0,  # No rounding for connected boxes
            )
            new_box.move_to(old_box.get_center())

            rank_transform_animations.append(
                Transform(old_box, new_box)
            )

        scene.play(
            *rank_transform_animations,
            run_time=duration
        )




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
            content = "Cross\nAttention\nTransformer"
            fill_color = MATERIAL_GREEN
            stroke_color = MATERIAL_GREEN_STROKE
        else:
            content = "Self\nAttention\nTransformer"
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
        show_pdf=False,
        show_probability_distribution=False,
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
            # self.output_dim = MonospaceText("N × 1", font_size=24, color=get_text_color())
            # self.output_dim.next_to(self.fc_layer, UP, buff=0.5)
            # self.add(self.output_dim)

            if show_pdf:
                # Simple token output to the right of FC
                self.pdf_token = Tokens(
                    num_tokens=1,
                    token_dim=1,
                    box_width=0.8,
                    box_height=2.0,
                    colors=[MATERIAL_BLUE],
                    stroke_colors=[MATERIAL_BLUE_STROKE],
                    label="Output PDF",
                    dim_label="N × 1",
                )
                self.pdf_token.scale(0.7)
                self.pdf_token.next_to(self.fc_layer, RIGHT, buff=1.5)

                # Arrow from FC to PDF token
                self.fc_to_pdf_arrow = Arrow(
                    self.fc_layer.get_right() + RIGHT * 0.2,
                    self.pdf_token.get_left() + LEFT * 0.2,
                    color=get_arrow_color(),
                    buff=0.1
                )

                self.add(self.fc_to_pdf_arrow, self.pdf_token)

            elif show_probability_distribution:
                # Full probability distribution visualization - match attention block height
                attention_height = self.attention.block.height
                self.prob_dist = ProbabilityDistribution(
                    num_visible=4,
                    total_items=10,
                    attention_height=attention_height
                )
                self.prob_dist.next_to(self.fc_layer, RIGHT, buff=1.5)
                # Align vertically with attention block
                self.prob_dist.align_to(self.attention, UP)

                # Arrow from FC to probability distribution
                self.fc_to_prob_arrow = Arrow(
                    self.fc_layer.get_right() + RIGHT * 0.2,
                    [self.prob_dist.get_left()[0] - 0.2, self.fc_layer.get_right()[1], 0],
                    color=get_arrow_color(),
                    buff=0.1
                )

                self.add(self.fc_to_prob_arrow, self.prob_dist)

        # Add positional encoding label
        # self.pos_encoding = Text("Positional Encoding", font_size=18, color=GRAY)
        # self.pos_encoding.next_to(self.attention, DOWN, buff=0.5)
        # self.add(self.pos_encoding)

        # Center the entire transformer
        self.move_to(ORIGIN)

    def animate_show_pdf(self, scene, duration=1.5):
        """Animate transformation from simple PDF token to full probability distribution.

        This method should be called when show_pdf=True to transform the simple token
        into the full probability distribution visualization.
        """
        if not hasattr(self, 'pdf_token'):
            raise ValueError("animate_show_pdf requires show_pdf=True when creating Transformer")

        # Create the full probability distribution at the same position
        attention_height = self.attention.block.height
        prob_dist = ProbabilityDistribution(
            num_visible=4,
            total_items=10,
            attention_height=attention_height
        )
        prob_dist.next_to(self.fc_layer, RIGHT, buff=1.5)
        prob_dist.align_to(self.attention, UP)

        # Animate the transformation
        scene.play(
            Transform(self.pdf_token, prob_dist),
            run_time=duration
        )

        # Remove old token and arrow from scene and self
        scene.remove(self.pdf_token)
        self.remove(self.pdf_token)

        # Store the new components
        self.prob_dist = prob_dist
        scene.add(self.prob_dist)
        self.add(self.prob_dist)

        # Animate sorting the probability distribution
        scene.wait(2)
        self.prob_dist.animate_sort_descending(scene, duration=duration)

        scene.wait(2)
        scene.wait(2)
