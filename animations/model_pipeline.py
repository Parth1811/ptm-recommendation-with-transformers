import numpy as np
from color_constants import (MATERIAL_BLUE, MATERIAL_ORANGE, MATERIAL_PURPLE,
                             MATERIAL_TEAL, MATERIAL_YELLOW, get_arrow_color,
                             get_stroke_color, get_text_color,
                             get_token_model_color)
from manim import *
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import ModelTokens


class ModelPipeline(VGroup):
    """Complete model processing pipeline visualization.

    Shows the detailed process from neural network to model token:
    1. Neural network with extracted parameters
    2. Clustering of parameters
    3. Reverse concatenation
    4. Padding/trimming to fixed size
    5. Positional encoding
    6. Model encoder
    7. Final model token
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track all components for animation
        self.all_components = VGroup()

        # ===== 1. NEURAL NETWORK WITH PARAMETERS =====
        self.model_network = NeuralNetwork(
            layers=[4, 3, 3, 4],
            node_radius=0.15,
            node_opacity=1,
            layer_spacing=0.8,
            abbreviated_hidden=True,
            abbreviated_spacing=0.8,
            show_labels=False,
        )
        self.model_network.scale(0.9)
        self.model_network.shift(LEFT * 6)

        # Parameter boxes below neural network
        self.param_boxes = self._create_parameter_boxes()
        self.param_boxes.next_to(self.model_network, DOWN, buff=0.4)
        self.param_boxes.align_to(self.model_network, LEFT)
        self.center_line = (self.model_network.get_center() + self.param_boxes.get_center()) * 0.5

        self.all_components.add(self.model_network, self.param_boxes)

        # ===== 2. CLUSTERING BOX =====
        self.clustering_box = RoundBox(
            "C\nL\nU\nS\nT\nE\nR\nI\nN\nG",
            width=1.0,
            height=3.5,
            fill_color=MATERIAL_TEAL,
            stroke_color=MATERIAL_TEAL,
            stroke_width=3,
            font_size=22,
            text_align="center",
        )
        self.clustering_box.next_to(self.model_network, RIGHT, buff=0.6)
        self.clustering_box.shift(UP * (self.center_line[1] - self.clustering_box.get_center()[1]))

        # Arrow to clustering (on center line)
        self.arrow1 = Arrow(
            np.array([self.model_network.get_right()[0], self.center_line[1], 0]),
            np.array([self.clustering_box.get_left()[0], self.center_line[1], 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        self.all_components.add(self.clustering_box, self.arrow1)

        # ===== 3. CLUSTERED PARAMETERS =====
        self.clustered_params = self._create_clustered_params()
        self.clustered_params.next_to(self.clustering_box, RIGHT, buff=0.6)
        self.clustered_params.align_to(self.model_network, UP)

        # Arrow to clustered params (on center line)
        self.arrow2 = Arrow(
            np.array([self.clustering_box.get_right()[0], self.center_line[1], 0]),
            np.array([self.clustered_params.get_left()[0], self.center_line[1], 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        self.all_components.add(self.clustered_params, self.arrow2)

        # ===== 4. COMBINING OPERATION =====
        # Create the combining box
        self.combining_box = RoundBox(
            "Combining",
            width=3.0,
            height=0.7,
            fill_color=MATERIAL_BLUE,
            stroke_color=MATERIAL_BLUE,
            stroke_width=2,
            font_size=18,
            text_align="center",
        )
        self.combining_box.next_to(self.clustered_params, DOWN, buff=0.5)
        self.combining_box.shift(DOWN * 0.8)
        self.combining_box.align_to(self.clustered_params, LEFT)

        # Clustered params representation (small boxes below combining)
        self.combined_boxes = VGroup()
        box_colors = [MATERIAL_PURPLE, get_stroke_color(), get_stroke_color(), MATERIAL_YELLOW]
        for i, color in enumerate(box_colors):
            if i == 1:
                # Add ".." in the middle
                ellipsis = MonospaceText("..", font_size=16, color=get_text_color())
                ellipsis.shift(RIGHT * (i - 1.5) * 0.4)
                self.combined_boxes.add(ellipsis)
            else:
                box = Rectangle(
                    width=0.4,
                    height=0.4,
                    fill_color=color,
                    fill_opacity=0.8,
                    stroke_color=color,
                    stroke_width=2,
                )
                offset = i if i <= 1 else i + 1  # Skip position 2 for ellipsis
                box.shift(RIGHT * (offset - 1.5) * 0.4)
                self.combined_boxes.add(box)

        self.combined_boxes.next_to(self.combining_box, DOWN, buff=0.3)

        self.all_components.add(self.combining_box, self.combined_boxes)

        # ===== 5. MODEL BOX =====
        # Model box (10000 x 1) - aligned on same center line as clustering
        self.model_box = RoundBox(
            "",
            width=0.7,
            height=3.5,
            fill_color=GRAY_B,
            stroke_color=WHITE,
            stroke_width=2,
            font_size=20,
            text_align="center",
        )
        self.model_box.next_to(self.combined_boxes, RIGHT, buff=1.5)
        self.model_box.shift(UP * (self.center_line[1] - self.model_box.get_center()[1]))

        self.model_box_label = MonospaceText("Model", font_size=16, color=get_text_color())
        self.model_box_label.next_to(self.model_box, UP, buff=0.2)

        # Dimension label for model box
        self.model_dim_label = MonospaceText("10000 × 1", font_size=16, color=get_text_color())
        self.model_dim_label.next_to(self.model_box, DOWN, buff=0.2)

        # Arrow from clustered params area to model box (on center line)
        self.arrow3 = Arrow(
            np.array([self.clustered_params.get_right()[0], self.center_line[1], 0]),
            np.array([self.model_box.get_left()[0], self.center_line[1], 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        self.all_components.add(self.model_box, self.model_box_label, self.model_dim_label, self.arrow3)

        # ===== 6. MODEL ENCODER (Triangular) =====
        self.model_encoder = self._create_encoder()
        self.model_encoder.next_to(self.model_box, RIGHT, buff=0.8)
        self.model_encoder.shift(UP * (self.center_line[1] - self.model_encoder.get_center()[1]))

        # Encoder label
        self.encoder_label = MonospaceText("Model\nEncoder", font_size=18,
                                           color=WHITE, line_spacing=0.8)
        self.encoder_label.move_to(self.model_encoder.get_center())

        # Arrow to encoder (on center line)
        self.arrow4 = Arrow(
            np.array([self.model_box.get_right()[0], self.center_line[1], 0]),
            np.array([self.model_encoder.get_left()[0], self.center_line[1], 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        self.all_components.add(self.model_encoder, self.encoder_label, self.arrow4)

        # ===== 7. MODEL TOKEN =====
        self.model_token = RoundBox(
            "",
            width=0.6,
            height=1.8,
            fill_color=get_token_model_color(),
            stroke_color=WHITE,
            stroke_width=2,
            font_size=16,
            text_align="center",
        )
        self.model_token.next_to(self.model_encoder, RIGHT, buff=0.8)
        self.model_token.shift(UP * (self.center_line[1] - self.model_token.get_center()[1]))

        self.model_token_label = MonospaceText("Model\nToken", font_size=18,
                                               color=WHITE, line_spacing=0.8)
        self.model_token_label.next_to(self.model_token, UP, buff=0.2)

        # Token dimension label
        self.token_dim_label = MonospaceText("512 × 1", font_size=16, color=get_text_color())
        self.token_dim_label.next_to(self.model_token, DOWN, buff=0.2)

        # Arrow to token (on center line)
        self.arrow5 = Arrow(
            np.array([self.model_encoder.get_right()[0], self.center_line[1], 0]),
            np.array([self.model_token.get_left()[0], self.center_line[1], 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        self.all_components.add(self.model_token, self.model_token_label, self.token_dim_label, self.arrow5)

        # Add all components to self
        self.add(self.all_components)

        # Center the entire pipeline
        self.move_to(ORIGIN)

    def _create_parameter_boxes(self):
        """Create parameter representation boxes below the network."""
        params = VGroup()

        # Define colors for each layer
        colors = [MATERIAL_YELLOW, get_stroke_color(), MATERIAL_PURPLE]
        labels = ["Input\nLayer", "Hidden\nLayers", "Output\nLayer"]

        # Positions aligned with network layers (considering abbreviated view)
        positions = [0, 1, 2, 3]  # Visual positions for 4 layers in abbreviated view

        for i, (pos, color, label) in enumerate(zip([0, 1.5, 3], colors[:3], labels)):
            # Create parameter box
            param_content = "w1\nw2\n..\n..\nb1\nb2\n.."
            param_box = RoundBox(
                param_content,
                width=0.6,
                height=1.8,
                fill_color=color,
                stroke_color=color,
                stroke_width=2,
                font_size=12,
                text_align="center",
            )
            param_box.shift(RIGHT * pos * 0.6)

            # Add layer label below
            layer_label = MonospaceText(label, font_size=14, color=color, line_spacing=0.7)
            layer_label.next_to(param_box, DOWN, buff=0.15)

            param_group = VGroup(param_box, layer_label)
            params.add(param_group)

        return params

    def _create_clustered_params(self):
        """Create the clustered parameter groups (20x1, 10x1, 10x1, 20x1)."""
        clusters = VGroup()

        # Define cluster specs: (content, dimension, color)
        cluster_specs = [
            ("w'1\nw'2\n..\n..\nw'20", "20x1", MATERIAL_YELLOW),
            ("w'1\nw'2\n..\n..\nw'10", "10x1", get_stroke_color()),
            ("w'1\nw'2\n..\n..\nw'10", "10x1", get_stroke_color()),
            ("w'1\nw'2\n..\n..\nw'20", "20x1", MATERIAL_PURPLE),
        ]

        for i, (content, dim, color) in enumerate(cluster_specs):
            # Create parameter box
            box = RoundBox(
                content,
                width=0.6,
                height=1.5,
                fill_color=color,
                stroke_color=color,
                stroke_width=2,
                font_size=10,
                text_align="center",
            )

            # Add dimension label
            dim_label = MonospaceText(dim, font_size=12, color=get_text_color())
            dim_label.next_to(box, DOWN, buff=0.1)

            cluster_group = VGroup(box, dim_label)
            cluster_group.shift(RIGHT * i * 0.9)

            clusters.add(cluster_group)

        return clusters

    def _create_encoder(self):
        """Create triangular/trapezoidal encoder shape."""
        # Create a trapezoid (wide on left, narrow on right)
        vertices = [
            LEFT * 0.5 + UP * 1.0,      # Top left
            RIGHT * 1.0 + UP * 0.6,     # Top right
            RIGHT * 1.0 + DOWN * 0.6,   # Bottom right
            LEFT * 0.5 + DOWN * 1.0,    # Bottom left
        ]

        encoder = Polygon(
            *vertices,
            fill_color=MATERIAL_TEAL,
            fill_opacity=0.8,
            stroke_color=MATERIAL_TEAL,
            stroke_width=3,
        )

        return encoder

    def animate_forward(self, scene, run_time=12):
        """Animate the forward pass through the model pipeline.

        Shows the complete transformation from neural network to model token:
        1. Network edges transform into parameter boxes
        2. Parameter boxes move to clustered positions
        3. Clustered parts move down to combined section
        4. Combined representation moves to encoder to make a token
        """
        time_unit = run_time / 12  # Divide into 12 steps (after removing commented sections)

        # Step 1: Animate neural network forward pass
        # self.model_network.animate_forward_pass(scene, run_time=time_unit * 3)
        # scene.wait(time_unit * 0.5)

        # Step 2: Transform network edges into parameter boxes
        # Create temporary edge highlights
        edge_highlights = VGroup()
        for edge_group in self.model_network.edge_groups:
            for edge in edge_group:
                highlight = edge.copy().set_stroke(YELLOW, width=3, opacity=0.8)
                edge_highlights.add(highlight)

        scene.add(edge_highlights)
        scene.play(
            edge_highlights.animate.set_stroke(opacity=1),
            run_time=time_unit * 0.8
        )

        # Transform edges into parameter boxes
        scene.play(
            *[edge_highlights[i].animate.move_to(self.param_boxes[i // (len(edge_highlights) // len(self.param_boxes))].get_center()).set_opacity(0)
              for i in range(len(edge_highlights))],
            *[param_group.animate.set_opacity(1) for param_group in self.param_boxes],
            *[Indicate(param_group, color=YELLOW, scale_factor=1.15) for param_group in self.param_boxes],
            run_time=time_unit * 1.5
        )
        scene.remove(edge_highlights)
        scene.wait(time_unit * 0.3)

        # Step 3: Highlight clustering box
        # scene.play(
        #     self.clustering_box.animate.scale(1.15).set_opacity(1),
        #     run_time=time_unit * 0.8
        # )
        # scene.play(
        #     self.clustering_box.animate.scale(1/1.15),
        #     run_time=time_unit * 0.5
        # )

        # Step 4: Move parameter boxes to clustered positions
        # Create copies of parameter boxes that will move
        param_copies = VGroup()
        for param_group in self.param_boxes:
            param_copy = param_group[0].copy()  # Just the box, not the label
            param_copies.add(param_copy)
            scene.add(param_copy)

        # Define which parameter maps to which cluster (yellow->cluster0, gray->clusters1&2, purple->cluster3)
        cluster_mapping = [0, 1, 3]  # Input->cluster0, Hidden->cluster1, Output->cluster3

        move_anims = []
        for i, param_copy in enumerate(param_copies):
            cluster_idx = cluster_mapping[i]
            target_pos = self.clustered_params[cluster_idx][0].get_center()  # Get box position from cluster
            move_anims.append(param_copy.animate.move_to(target_pos).scale(0.8))

        scene.play(
            *move_anims,
            Circumscribe(self.clustering_box, color=YELLOW, stroke_width=3, buff=0.1),
            run_time=time_unit * 1.5
        )

        # Fade out copies and fade in actual clustered params
        scene.play(
            *[param_copy.animate.set_opacity(0) for param_copy in param_copies],
            *[cluster.animate.set_opacity(1) for cluster in self.clustered_params],
            *[Indicate(cluster, color=YELLOW, scale_factor=1.1) for cluster in self.clustered_params],
            run_time=time_unit * 1.0
        )
        scene.remove(*param_copies)
        scene.wait(time_unit * 0.3)

        # Step 5: Move clustered parts down to combined section
        # Highlight combining box
        # scene.play(
        #     self.combining_box.animate.set_opacity(1).scale(1.1),
        #     run_time=time_unit * 0.8
        # )
        # scene.play(
        #     self.combining_box.animate.scale(1/1.1),
        #     run_time=time_unit * 0.5
        # )

        # Create copies of clustered params that will move down
        cluster_copies = VGroup()
        for cluster in self.clustered_params:
            cluster_copy = cluster[0].copy()  # Just the box
            cluster_copy.scale(0.5)  # Scale down for small boxes
            cluster_copies.add(cluster_copy)
            scene.add(cluster_copy)

        # Move cluster copies to combined boxes positions
        move_down_anims = []
        # Map to combined_boxes indices (skip ellipsis at index 1)
        box_indices = [3, 2, 2, 0]  # Purple, gray, yellow in combined_boxes (skipping ellipsis)
        for i, cluster_copy in enumerate(cluster_copies):
            if i < len(box_indices):
                target_box_idx = box_indices[i]
                target_pos = self.combined_boxes[target_box_idx].get_center()
                move_down_anims.append(cluster_copy.animate.move_to(target_pos))

        scene.play(
            *move_down_anims,
            Circumscribe(self.combining_box, color=YELLOW, stroke_width=3, buff=0.1),
            run_time=time_unit * 1.5
        )

        # Fade out copies and fade in combined boxes
        scene.play(
            *[cluster_copy.animate.set_opacity(0) for cluster_copy in cluster_copies],
            self.combined_boxes.animate.set_opacity(1),
            run_time=time_unit * 0.8
        )
        scene.remove(*cluster_copies)
        scene.wait(time_unit * 0.3)

        # Step 6: Move combined representation to model box through encoder
        # Create a copy of combined boxes that will move
        combined_copy = self.combined_boxes.copy()
        scene.add(combined_copy)

        # Move to model box
        scene.play(
            combined_copy.animate.move_to(self.model_box.get_center()).scale(0.5).set_opacity(0.5),
            self.model_box.animate.set_opacity(1),
            self.model_dim_label.animate.set_opacity(1),
            Indicate(self.model_box, color=YELLOW, scale_factor=1.1),
            run_time=time_unit * 1.5
        )

        # Fade out combined copy
        scene.play(
            combined_copy.animate.set_opacity(0),
            run_time=time_unit * 0.5
        )
        scene.remove(combined_copy)

        # Pulse model box
        # scene.play(
        #     self.model_box.animate.scale(1.1),
        #     run_time=time_unit * 0.5
        # )
        # scene.play(
        #     self.model_box.animate.scale(1/1.1),
        #     run_time=time_unit * 0.5
        # )

        # Step 7: Highlight encoder and create token
        # scene.play(
        #     self.model_encoder.animate.scale(1.15).set_opacity(1),
        #     self.encoder_label.animate.scale(1.15).set_opacity(1),
        #     run_time=time_unit * 0.8
        # )
        # scene.play(
        #     self.model_encoder.animate.scale(1/1.15),
        #     self.encoder_label.animate.scale(1/1.15),
        #     run_time=time_unit * 0.5
        # )
        scene.play(
            Indicate(self.model_encoder, color=YELLOW, scale_factor=1.1),
            run_time=time_unit * 0.8
        )

        # Create a small box that moves from encoder to token
        token_creator = Rectangle(
            width=0.2,
            height=0.6,
            fill_color=get_token_model_color(),
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2
        )
        token_creator.move_to(self.model_encoder.get_center())
        scene.add(token_creator)

        # Move to token position
        scene.play(
            token_creator.animate.move_to(self.model_token.get_center()).scale(3),
            run_time=time_unit * 1.2
        )

        # Fade out creator and fade in actual token
        scene.play(
            token_creator.animate.set_opacity(0),
            self.model_token.animate.set_opacity(1),
            self.token_dim_label.animate.set_opacity(1),
            Indicate(self.model_token, color=YELLOW, scale_factor=1.2),
            run_time=time_unit * 1.0
        )
        scene.remove(token_creator)
