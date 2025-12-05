from color_constants import (MATERIAL_BLUE, MATERIAL_BLUE_STROKE,
                             MATERIAL_DARK_GRAY, MATERIAL_DARK_GRAY_STROKE,
                             MATERIAL_GRAY, MATERIAL_GRAY_STROKE,
                             MATERIAL_MINT, MATERIAL_MINT_STROKE,
                             MATERIAL_YELLOW, MATERIAL_YELLOW_STROKE,
                             get_arrow_color, get_clustering_color,
                             get_encoder_image_color, get_encoder_text_color,
                             get_sample_box_color, get_sampling_color,
                             get_stroke_color, get_text_color)
from manim import *
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import DatasetTokens


class DatasetPipeline(VGroup):
    """Complete dataset processing pipeline visualization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unstructured dataset
        self.unstructured = self._create_data_grid("Unstructured\nDataset", 4, 4, seed=42, labels=["A", "B", "C"])
        self.add(self.unstructured)

        # Clustering box
        self.clustering1 = self._create_vertical_box("C\nL\nU\nS\nT\nE\nR\nI\nN\nG",
                                                      fill_color=MATERIAL_MINT,
                                                      stroke_color=MATERIAL_MINT_STROKE)
        self.clustering1.next_to(self.unstructured, RIGHT, buff=0.5)
        self.add(self.clustering1)

        # Arrow
        arrow1 = Arrow(
            self.unstructured.get_right(),
            self.clustering1.get_left(),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow1)

        # Structured dataset
        self.structured1 = self._create_data_grid("A", 2, 2, seed=100, labels=["A"])
        self.structured1.next_to(self.clustering1, RIGHT, buff=0.5)
        self.structured1.shift(UP * 0.9)
        self.add(self.structured1)

        self.structured2 = self._create_data_grid("B", 2, 2, seed=101, labels=["B"])
        self.structured2.next_to(self.structured1, RIGHT, buff=0.5)
        self.add(self.structured2)

        self.structured3 = self._create_data_grid("C", 2, 2, seed=102, labels=["C"])
        self.structured3.next_to(self.clustering1, RIGHT, buff=0.5)
        self.structured3.shift(RIGHT + DOWN * 0.9)
        self.add(self.structured3)

        center_y = self.clustering1.get_center()[1]

        # Arrow
        arrow2 = Arrow(
            self.clustering1.get_right(),
            np.array([self.structured1.get_left()[0] + 0.4, center_y, 0]),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow2)

        # Sampling box
        self.sampling = self._create_vertical_box("S\nA\nM\nP\nL\nI\nN\nG",
                                                   fill_color=MATERIAL_BLUE,
                                                   stroke_color=MATERIAL_BLUE_STROKE)
        self.sampling.next_to(self.structured2, RIGHT, buff=0.5)
        self.sampling.shift(UP * (center_y - self.sampling.get_center()[1]))
        self.add(self.sampling)

        # Arrow
        arrow3 = Arrow(
            np.array([self.structured2.get_right()[0] - 0.4, center_y, 0]),
            self.sampling.get_left(),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow3)

        # Representative samples
        self.samples = VGroup()
        sample_labels = ["A", "B", "C", "..."]
        for i, label in enumerate(sample_labels):
            # Use label-based color, or teal for ellipsis
            if label == "...":
                sample_fill = MATERIAL_DARK_GRAY
                sample_stroke = MATERIAL_DARK_GRAY_STROKE
            else:
                sample_fill, sample_stroke = self._get_color_for_label(label)

            sample = RoundBox(
                content=label,
                width=0.8,
                height=0.8,
                fill_color=sample_fill,
                stroke_color=sample_stroke,
                stroke_width=3,
                font_size=24,
            )
            sample.shift(DOWN * i * 1.0)
            self.samples.add(sample)

        self.samples.next_to(self.sampling, RIGHT, buff=1.5)
        self.add(self.samples)

        # Arrow
        arrow4 = Arrow(
            self.sampling.get_right(),
            self.samples.get_left(),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow4)

        # Encoder as Neural Network
        self.encoder_network = NeuralNetwork(
            layers=[4, 3, 2],
            layer_spacing=1.2,
            node_radius=0.12,
            node_opacity=0.8,
            show_labels=False,
        )
        self.encoder_network.scale(0.6)
        self.encoder_network.next_to(self.samples, RIGHT, buff=1.5)
        self.encoder_network.shift(LEFT * 0.2)
        self.add(self.encoder_network)

        # Encoder label
        self.encoder_label = MonospaceText("Encoder", font_size=18, color=get_text_color())
        self.encoder_label.next_to(self.encoder_network, DOWN, buff=0.3)
        self.add(self.encoder_label)

        # Arrows from samples to encoder
        arrow5 = Arrow(
            self.samples.get_right() + RIGHT * 0.2,
            self.encoder_network.get_left(),
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow5)

        # Label
        self.samples_label = MonospaceText("Representative\nSamples", font_size=20, color=get_text_color())
        self.samples_label.next_to(self.samples, DOWN, buff=0.5)
        self.add(self.samples_label)

        # Dataset tokens output as actual tokens
        self.dataset_tokens = DatasetTokens(
            num_samples=3,
            colors=[MATERIAL_YELLOW, MATERIAL_MINT, MATERIAL_BLUE],
            stroke_colors=[MATERIAL_YELLOW_STROKE, MATERIAL_MINT_STROKE, MATERIAL_BLUE_STROKE],
            abbreviated=True
        )
        self.dataset_tokens.scale(0.6)
        self.dataset_tokens.next_to(self.encoder_network, RIGHT, buff=1.2)
        self.dataset_tokens.shift(LEFT * 0.2)
        self.add(self.dataset_tokens)

        # Add arrow from encoder to tokens
        arrow7 = Arrow(
            self.encoder_network.get_right(),
            self.dataset_tokens.get_left(),
            buff=0.2,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color()
        )
        self.add(arrow7)

        # Center the entire pipeline
        self.move_to(ORIGIN)

    def _get_color_for_label(self, label):
        """Return (fill_color, stroke_color) tuple based on dataset label."""
        color_map = {
            "A": (MATERIAL_YELLOW, MATERIAL_YELLOW_STROKE),     # Yellow/gold for A (Q3)
            "B": (MATERIAL_MINT, MATERIAL_MINT_STROKE),         # Green/teal for B
            "C": (MATERIAL_BLUE, MATERIAL_BLUE_STROKE),         # Blue for C (Q1, Q2)
        }
        return color_map.get(label, (MATERIAL_BLUE, MATERIAL_BLUE_STROKE))  # Default to blue

    def _create_data_grid(self, label, rows, cols, seed=42, labels=None):
        """Create a scattered grid of small rounded boxes representing data.

        Args:
            label: Label text below the grid
            rows: Number of rows
            cols: Number of columns
            seed: Random seed for reproducibility
            labels: List of labels to use (e.g., ["A", "B", "C"]). If None, uses sequential letters.
        """
        import random

        # Set seed for reproducibility
        random.seed(seed)

        group = VGroup()

        grid = VGroup()
        box_width = 0.5
        box_height = 0.35

        # Calculate base spacing
        base_spacing_x = box_width + 0.15
        base_spacing_y = box_height + 0.15

        # Create total number of boxes
        total_boxes = rows * cols

        # Create positions with some randomization
        positions = []
        for i in range(rows):
            for j in range(cols):
                # Base position
                x = j * base_spacing_x
                y = -i * base_spacing_y

                # Add random offset for scattered effect
                x_offset = random.uniform(-0.1, 0.1)
                y_offset = random.uniform(-0.1, 0.1)

                positions.append((x + x_offset, y + y_offset))

        # Shuffle positions for more scattered look
        random.shuffle(positions)

        # Prepare labels to distribute
        if labels is None:
            labels = [chr(ord("A") + i) for i in range(3)]

        # Create a pool of content with labels repeated
        content_pool = []
        num_labels = len(labels)
        num_with_labels = min(total_boxes // 2, num_labels * 2)  # Distribute labels across half the boxes

        for i in range(num_with_labels):
            content_pool.append(labels[i % num_labels])

        # Fill rest with ".." or empty
        for i in range(total_boxes - num_with_labels):
            if random.random() > 0.5:
                content_pool.append("..")
            else:
                content_pool.append("")

        # Shuffle content pool
        random.shuffle(content_pool)

        # Create boxes at shuffled positions
        for idx, (x, y) in enumerate(positions):
            content = content_pool[idx]

            if content in labels:
                font_size = 16
            elif content == "..":
                font_size = 14
            else:
                font_size = 14

            if content:
                # Determine box color based on grid label
                if label in ["A", "B", "C"]:
                    box_fill, box_stroke = self._get_color_for_label(label)
                elif content in ["A", "B", "C"]:
                    box_fill, box_stroke = self._get_color_for_label(content)
                else:
                    box_fill, box_stroke = MATERIAL_BLUE, MATERIAL_BLUE_STROKE

                box = RoundBox(
                    content=content,
                    width=box_width,
                    height=box_height,
                    fill_color=box_fill,
                    stroke_color=box_stroke,
                    stroke_width=2,
                    font_size=font_size,
                    corner_radius=0.05,
                )
            else:
                if label in ["A", "B", "C"]:
                    box_fill, box_stroke = self._get_color_for_label(label)
                else:
                    box_fill, box_stroke = MATERIAL_GRAY, MATERIAL_GRAY_STROKE

                box = RoundBox(
                    content="",
                    width=box_width,
                    height=box_height,
                    fill_color=box_fill,
                    stroke_color=box_stroke,
                    corner_radius=0.05,
                    stroke_width=2,
                )

            box.shift(RIGHT * x + UP * y)
            grid.add(box)

        # Center the grid
        grid.move_to(ORIGIN)

        label_text = MonospaceText(label, font_size=20, color=get_text_color())
        label_text.next_to(grid, DOWN, buff=0.3)

        group.add(grid, label_text)
        return group

    def _create_vertical_box(self, text, fill_color=None, stroke_color=None):
        """Create a vertical colored box with text."""
        if fill_color is None:
            fill_color = MATERIAL_MINT  # Default to teal
        if stroke_color is None:
            stroke_color = fill_color  # Default stroke to match fill

        return RoundBox(
            content=text,
            width=1.0,
            height=4.5,
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=3,
            font_size=28,
            text_align="center",
        )

    def animate_forward(self, scene, run_time=8):
        """Animate the forward pass through the dataset pipeline.

        Shows:
        1. Unstructured data boxes rearranging into structured clusters
        2. Representative samples being extracted from each cluster
        3. Samples passing through the encoder to become dataset tokens
        """
        # Step 1: Highlight and pulse clustering box
        scene.play(
            self.clustering1.animate.set_opacity(1),
            run_time=run_time * 0.1
        )

        # Step 2: Animate unstructured boxes rearranging into structured groups
        # Get all boxes from unstructured grid (first child is the grid VGroup)
        unstructured_boxes = self.unstructured[0]

        # Separate boxes by their labels
        boxes_with_A = []
        boxes_with_B = []
        boxes_with_C = []
        other_boxes = []

        for box in unstructured_boxes:
            if hasattr(box, 'content'):
                content = box.content.text if hasattr(box.content, 'text') else str(box.content)
                if content == "A":
                    boxes_with_A.append(box)
                elif content == "B":
                    boxes_with_B.append(box)
                elif content == "C":
                    boxes_with_C.append(box)
                else:
                    other_boxes.append(box)
            else:
                other_boxes.append(box)

        # Get target positions from structured grids
        structured1_boxes = self.structured1[0]
        structured2_boxes = self.structured2[0]
        structured3_boxes = self.structured3[0]

        # Create movement animations
        move_anims = []

        # Move A boxes to structured1
        for idx, box in enumerate(boxes_with_A[:len(structured1_boxes)]):
            if idx < len(structured1_boxes):
                target_pos = structured1_boxes[idx].get_center()
                move_anims.append(box.animate.move_to(target_pos))

        # Move B boxes to structured2
        for idx, box in enumerate(boxes_with_B[:len(structured2_boxes)]):
            if idx < len(structured2_boxes):
                target_pos = structured2_boxes[idx].get_center()
                move_anims.append(box.animate.move_to(target_pos))

        # Move C boxes to structured3
        for idx, box in enumerate(boxes_with_C[:len(structured3_boxes)]):
            if idx < len(structured3_boxes):
                target_pos = structured3_boxes[idx].get_center()
                move_anims.append(box.animate.move_to(target_pos))

        # Fade out other boxes
        fade_anims = [box.animate.set_opacity(0.1) for box in other_boxes]

        # Execute all movements simultaneously
        scene.play(
            *move_anims,
            *fade_anims,
            run_time=run_time * 0.3
        )

        # Fade out moved boxes and fade in structured grids
        scene.play(
            *[box.animate.set_opacity(0) for box in boxes_with_A + boxes_with_B + boxes_with_C],
            self.structured1[0].animate.set_opacity(1),
            self.structured2[0].animate.set_opacity(1),
            self.structured3[0].animate.set_opacity(1),
            run_time=run_time * 0.1
        )

        # Step 3: Highlight sampling box
        scene.play(
            self.sampling.animate.set_opacity(1),
            run_time=run_time * 0.1
        )

        # Step 4: Extract one box from each structured grid to become representative samples
        # Create temporary boxes that will move from structured to samples
        temp_boxes = []

        # Get first labeled box from each structured grid
        for structured_grid in [self.structured1, self.structured2, self.structured3]:
            grid_boxes = structured_grid[0]
            # Find first box with content
            for box in grid_boxes:
                if hasattr(box, 'content'):
                    temp_box = box.copy()
                    temp_boxes.append(temp_box)
                    scene.add(temp_box)
                    break

        # Animate boxes moving to representative samples positions
        if len(temp_boxes) >= 3:
            scene.play(
                temp_boxes[0].animate.move_to(self.samples[0].get_center()),
                temp_boxes[1].animate.move_to(self.samples[1].get_center()),
                temp_boxes[2].animate.move_to(self.samples[2].get_center()),
                run_time=run_time * 0.2
            )

            # Fade out temp boxes and fade in actual sample boxes
            scene.play(
                *[temp_box.animate.set_opacity(0) for temp_box in temp_boxes],
                *[sample.animate.set_opacity(1) for sample in self.samples[:3]],
                run_time=run_time * 0.1
            )

            # Remove temp boxes
            for temp_box in temp_boxes:
                scene.remove(temp_box)

        # Step 5: Highlight representative samples
        scene.play(
            self.samples.animate.set_opacity(1),
            self.samples_label.animate.set_opacity(1),
            run_time=run_time * 0.1
        )

        # Step 6: Animate all three boxes A, B, C going into the encoder network
        # Create copies of sample boxes to animate
        sample_copies = []
        for i in range(3):  # A, B, C
            sample_copy = self.samples[i].copy()
            sample_copies.append(sample_copy)
            scene.add(sample_copy)

        # Process each sample one by one
        for idx, sample_copy in enumerate(sample_copies):
            # Highlight the sample
            scene.play(
                sample_copy.animate.scale(1.2).set_opacity(1),
                run_time=run_time * 0.03
            )
            scene.play(
                sample_copy.animate.scale(1/1.2),
                run_time=run_time * 0.03
            )

            # Move sample into the encoder network (to first layer)
            encoder_input_pos = self.encoder_network.layer_groups[0].get_center()
            scene.play(
                sample_copy.animate.move_to(encoder_input_pos).scale(0.3),
                run_time=run_time * 0.08
            )

            # Fade out the sample box as it enters the network
            scene.play(
                sample_copy.animate.set_opacity(0),
                run_time=run_time * 0.03
            )

            # Step 7: Animate forward pass through the encoder network
            self.encoder_network.animate_forward_pass(scene, run_time=run_time * 0.12, flow_color=YELLOW)

            # Step 8: Show corresponding dataset token appearing
            # Get the corresponding token (first 3 are actual tokens, 4th is "...")
            token_idx = idx
            token = self.dataset_tokens.token_boxes[token_idx]

            # Create a small version at encoder output
            token_copy = token.copy()
            token_copy.scale(0.5)
            token_copy.move_to(self.encoder_network.layer_groups[-1].get_center())
            token_copy.set_opacity(0)
            scene.add(token_copy)

            # Fade in at encoder output and move to final position
            scene.play(
                token_copy.animate.set_opacity(1),
                run_time=run_time * 0.03
            )

            scene.play(
                token_copy.animate.move_to(token.get_center()).scale(2),
                run_time=run_time * 0.08
            )

            # Fade out copy and fade in actual token
            scene.play(
                token_copy.animate.set_opacity(0),
                token.animate.set_opacity(1),
                run_time=run_time * 0.03
            )

            scene.remove(token_copy)
            scene.remove(sample_copy)

        # Show the abbreviated "..." token and labels
        if len(self.dataset_tokens.token_boxes) > 3:
            scene.play(
                self.dataset_tokens.token_boxes[3].animate.set_opacity(1),
                run_time=run_time * 0.05
            )

        # Fade in dimension label and main label
        scene.play(
            self.dataset_tokens.dim_label.animate.set_opacity(1),
            run_time=run_time * 0.05
        )

        if hasattr(self.dataset_tokens, 'label'):
            scene.play(
                self.dataset_tokens.label.animate.set_opacity(1),
                run_time=run_time * 0.05
            )

        # Final pulse on all tokens
        scene.play(
            self.dataset_tokens.animate.scale(1.1),
            run_time=run_time * 0.03
        )
        scene.play(
            self.dataset_tokens.animate.scale(1/1.1),
            run_time=run_time * 0.03
        )
