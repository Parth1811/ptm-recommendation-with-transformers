"""Visualization of model layer clustering process.

Shows raw parameter values from a model layer alongside their clustered centers.
"""

import numpy as np
from manim import *
from color_constants import (
    MATERIAL_BLUE, MATERIAL_BLUE_STROKE,
    MATERIAL_MINT, MATERIAL_MINT_STROKE,
    MATERIAL_PURPLE, MATERIAL_PURPLE_STROKE,
    MATERIAL_GRAY, MATERIAL_GRAY_STROKE,
    get_text_color, get_stroke_color
)
from monospace_text import MonospaceText
from round_box import RoundBox

import sys
import os
sys.path.append('..')

# Import torch and transformers
import torch
from transformers import AutoModel, AutoConfig
from sklearn.cluster import KMeans


class LayerClusteringVisualization(Scene):
    """Visualize raw layer values and their clustered centers side-by-side."""

    def construct(self):
        # Configuration
        model_id = "google/vit-base-patch16-224"  # Small ViT model
        n_clusters = 64
        layer_name = "encoder.layer.0.attention.attention.query.weight"

        # Title
        title = MonospaceText(
            "Layer Clustering: Raw vs Centers",
            font_size=36,
            color=get_text_color()
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)
        self.wait(0.5)

        # Load model and extract layer
        loading_text = MonospaceText(
            f"Loading {model_id}...",
            font_size=24,
            color=get_text_color()
        )
        loading_text.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(loading_text))

        # Actually load the model
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval()
        state_dict = model.state_dict()

        # Get the first attention layer (usually has good variation)
        available_keys = [k for k in state_dict.keys() if 'weight' in k]
        if layer_name not in available_keys:
            layer_name = available_keys[0]  # Fallback to first weight layer

        layer_tensor = state_dict[layer_name].detach().cpu().numpy().flatten()

        # Update loading text
        layer_info = MonospaceText(
            f"Layer: {layer_name[:40]}...\nShape: {state_dict[layer_name].shape}",
            font_size=20,
            color=get_text_color()
        )
        layer_info.next_to(title, DOWN, buff=0.5)
        self.play(Transform(loading_text, layer_info))
        self.wait(1)

        # Sample subset of values for visualization (too many to show all)
        sample_size = 1000
        sampled_values = np.random.choice(layer_tensor, size=sample_size, replace=False)

        # Perform clustering using sklearn KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sampled_values.reshape(-1, 1))
        cluster_centers = np.sort(kmeans.cluster_centers_.flatten())[::-1]  # Sort descending

        # Create visualization
        viz_label = MonospaceText(
            "Raw Distribution + Cluster Centers",
            font_size=28,
            color=get_text_color()
        )
        viz_label.shift(UP * 2.5)

        # Create histogram with axes
        histogram_viz = self._create_histogram_with_axes(
            sampled_values,
            cluster_centers,
            width=10,
            height=5,
            bins=50
        )
        histogram_viz.shift(DOWN * 0.3)

        # Create statistics panels
        raw_stats = self._create_stats_text(sampled_values, "Raw Values")
        raw_stats.next_to(histogram_viz, DOWN, buff=0.4)
        raw_stats.shift(LEFT * 3)

        cluster_stats = self._create_stats_text(cluster_centers, "Cluster Centers")
        cluster_stats.next_to(histogram_viz, DOWN, buff=0.4)
        cluster_stats.shift(RIGHT * 3)

        compression_text = MonospaceText(
            f"Compression: {len(sampled_values)} → {n_clusters} values",
            font_size=20,
            color=MATERIAL_MINT
        )
        compression_text.next_to(cluster_stats, DOWN, buff=0.3)

        # Create legend
        legend = self._create_legend()
        legend.to_corner(UR, buff=0.5)

        # Animate the creation
        self.play(FadeOut(loading_text))
        self.play(Write(viz_label), run_time=1)

        # Create histogram bars first
        self.play(Create(histogram_viz[0]), run_time=1)  # Border
        self.play(Create(histogram_viz[1]), run_time=2)  # Histogram bars
        self.play(Create(histogram_viz[2]), run_time=0.5)  # Axes

        # Then add cluster centers as overlay
        self.play(Create(histogram_viz[3]), run_time=2)  # Cluster center markers

        # Add labels, stats, and legend
        self.play(
            FadeIn(raw_stats),
            FadeIn(cluster_stats),
            FadeIn(compression_text),
            FadeIn(legend),
            run_time=1
        )

        # Hold the final view
        self.wait(3)

        # Fade out
        self.play(
            FadeOut(viz_label),
            FadeOut(histogram_viz),
            FadeOut(raw_stats),
            FadeOut(cluster_stats),
            FadeOut(compression_text),
            FadeOut(legend),
            FadeOut(title)
        )

    def _create_histogram_with_axes(self, values, cluster_centers, width, height, bins=50):
        """Create a histogram with axes showing raw values and cluster centers as overlay."""
        # Calculate histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1

        # Get value range
        val_min = min(values.min(), cluster_centers.min())
        val_max = max(values.max(), cluster_centers.max())
        val_range = val_max - val_min

        # Create bars for histogram
        bars = VGroup()
        bar_width = width / bins

        for i, count in enumerate(counts):
            bar_height = (count / max_count) * height
            bar = Rectangle(
                width=bar_width * 0.95,
                height=bar_height,
                fill_color=MATERIAL_BLUE,
                fill_opacity=0.6,
                stroke_color=MATERIAL_BLUE_STROKE,
                stroke_width=1
            )
            # Position bar
            x_pos = -width/2 + (i + 0.5) * bar_width
            y_pos = -height/2 + bar_height/2
            bar.move_to([x_pos, y_pos, 0])
            bars.add(bar)

        # Create border
        border = Rectangle(
            width=width,
            height=height,
            stroke_color=get_stroke_color(),
            stroke_width=3,
            fill_opacity=0
        )

        # Create axes with labels
        axes = self._create_axes(width, height, val_min, val_max, max_count)

        # Create cluster center markers (as vertical lines and dots)
        cluster_markers = VGroup()
        for center in cluster_centers:
            # Map center value to x position
            normalized_pos = (center - val_min) / val_range
            x_pos = -width/2 + normalized_pos * width

            # Create vertical line
            line = Line(
                start=[x_pos, -height/2, 0],
                end=[x_pos, height/2, 0],
                stroke_color=MATERIAL_MINT,
                stroke_width=3,
                stroke_opacity=0.8
            )

            # Create dot at top
            dot = Dot(
                point=[x_pos, height/2, 0],
                radius=0.08,
                color=MATERIAL_MINT,
                stroke_color=MATERIAL_MINT_STROKE,
                stroke_width=2
            )

            cluster_markers.add(line, dot)

        return VGroup(border, bars, axes, cluster_markers)

    def _create_axes(self, width, height, val_min, val_max, max_count):
        """Create x and y axes with labels."""
        axes_group = VGroup()

        # X-axis
        x_axis = Line(
            start=[-width/2, -height/2, 0],
            end=[width/2, -height/2, 0],
            stroke_color=get_stroke_color(),
            stroke_width=2
        )
        axes_group.add(x_axis)

        # Y-axis
        y_axis = Line(
            start=[-width/2, -height/2, 0],
            end=[-width/2, height/2, 0],
            stroke_color=get_stroke_color(),
            stroke_width=2
        )
        axes_group.add(y_axis)

        # X-axis labels (value range)
        x_label_positions = [0, 0.25, 0.5, 0.75, 1.0]
        for pos in x_label_positions:
            val = val_min + pos * (val_max - val_min)
            x_pos = -width/2 + pos * width
            label = MonospaceText(
                f"{val:.2f}",
                font_size=14,
                color=get_text_color()
            )
            label.move_to([x_pos, -height/2 - 0.3, 0])
            axes_group.add(label)

        # X-axis title
        x_title = MonospaceText("Parameter Value", font_size=18, color=get_text_color())
        x_title.next_to(x_axis, DOWN, buff=0.6)
        axes_group.add(x_title)

        # Y-axis labels (count)
        y_label_positions = [0, 0.5, 1.0]
        for pos in y_label_positions:
            count = int(pos * max_count)
            y_pos = -height/2 + pos * height
            label = MonospaceText(
                f"{count}",
                font_size=14,
                color=get_text_color()
            )
            label.move_to([-width/2 - 0.5, y_pos, 0])
            axes_group.add(label)

        # Y-axis title
        y_title = MonospaceText("Count", font_size=18, color=get_text_color())
        y_title.next_to(y_axis, LEFT, buff=0.8)
        y_title.rotate(PI/2)
        axes_group.add(y_title)

        return axes_group

    def _create_legend(self):
        """Create a legend for the visualization."""
        legend_group = VGroup()

        # Legend background
        bg = Rectangle(
            width=3,
            height=1.2,
            fill_color=MATERIAL_GRAY,
            fill_opacity=0.1,
            stroke_color=get_stroke_color(),
            stroke_width=2
        )
        legend_group.add(bg)

        # Raw values indicator (blue bar)
        raw_bar = Rectangle(
            width=0.3,
            height=0.15,
            fill_color=MATERIAL_BLUE,
            fill_opacity=0.6,
            stroke_color=MATERIAL_BLUE_STROKE,
            stroke_width=1
        )
        raw_bar.shift(LEFT * 0.8 + UP * 0.25)
        raw_text = MonospaceText("Raw Values", font_size=14, color=get_text_color())
        raw_text.next_to(raw_bar, RIGHT, buff=0.2)

        # Cluster centers indicator (green line)
        cluster_line = Line(
            start=[0, 0, 0],
            end=[0.3, 0, 0],
            stroke_color=MATERIAL_MINT,
            stroke_width=3
        )
        cluster_line.shift(LEFT * 0.8 + DOWN * 0.25)
        cluster_dot = Dot(
            point=[0.15, -0.25, 0],
            radius=0.05,
            color=MATERIAL_MINT,
            stroke_color=MATERIAL_MINT_STROKE,
            stroke_width=2
        )
        cluster_dot.shift(LEFT * 0.8)
        cluster_text = MonospaceText("Cluster Centers", font_size=14, color=get_text_color())
        cluster_text.next_to(cluster_line, RIGHT, buff=0.2)

        legend_group.add(raw_bar, raw_text, cluster_line, cluster_dot, cluster_text)

        return legend_group

    def _create_histogram(self, values, width, height, fill_color, stroke_color, bins=30):
        """Create a histogram visualization of the values."""
        # Calculate histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1

        # Create bars
        bars = VGroup()
        bar_width = width / bins

        for i, count in enumerate(counts):
            bar_height = (count / max_count) * height
            bar = Rectangle(
                width=bar_width * 0.9,  # Small gap between bars
                height=bar_height,
                fill_color=fill_color,
                fill_opacity=0.7,
                stroke_color=stroke_color,
                stroke_width=1
            )
            # Position bar
            x_pos = -width/2 + (i + 0.5) * bar_width
            y_pos = -height/2 + bar_height/2
            bar.move_to([x_pos, y_pos, 0])
            bars.add(bar)

        # Create container with border
        border = Rectangle(
            width=width,
            height=height,
            stroke_color=get_stroke_color(),
            stroke_width=2,
            fill_opacity=0
        )

        histogram = VGroup(border, bars)
        return histogram

    def _create_stats_text(self, values, label):
        """Create statistics text for a set of values."""
        stats_text = MonospaceText(
            f"{label}: n={len(values)}\n"
            f"mean={np.mean(values):.4f}\n"
            f"std={np.std(values):.4f}\n"
            f"min={np.min(values):.4f}\n"
            f"max={np.max(values):.4f}",
            font_size=16,
            color=get_text_color()
        )
        return stats_text


class LayerClusteringDetailed(Scene):
    """More detailed visualization showing the clustering process step-by-step."""

    def construct(self):
        # Configuration
        model_id = "google/vit-base-patch16-224"
        n_clusters = 16  # Fewer for clearer visualization

        # Title
        title = MonospaceText(
            "Layer Clustering Process",
            font_size=36,
            color=get_text_color()
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)

        # Step 1: Load model
        step1 = MonospaceText("Step 1: Load Model Layer", font_size=24, color=get_text_color())
        step1.next_to(title, DOWN, buff=0.5)
        self.play(Write(step1))

        # Load model directly
        model = AutoModel.from_pretrained(model_id).eval()
        state_dict = model.state_dict()

        # Get first weight layer
        layer_name = [k for k in state_dict.keys() if 'weight' in k][0]
        layer_tensor = state_dict[layer_name].detach().cpu().numpy().flatten()

        # Sample for visualization
        sample_size = 500
        sampled_values = np.random.choice(layer_tensor, size=sample_size, replace=False)

        # Show as scatter plot
        scatter_dots = VGroup()
        for i, val in enumerate(sampled_values[:100]):  # Show first 100
            dot = Dot(
                point=[val * 3, np.random.uniform(-2, 2), 0],
                radius=0.03,
                color=MATERIAL_BLUE
            )
            scatter_dots.add(dot)

        self.play(FadeIn(scatter_dots), run_time=2)
        self.wait(1)

        # Step 2: Apply K-means
        step2 = MonospaceText("Step 2: Apply K-Means Clustering", font_size=24, color=get_text_color())
        self.play(Transform(step1, step2))

        # Perform clustering using sklearn KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sampled_values.reshape(-1, 1))
        cluster_centers = np.sort(kmeans.cluster_centers_.flatten())[::-1]

        # Show cluster centers
        center_dots = VGroup()
        for center in cluster_centers:
            dot = Dot(
                point=[center * 3, 0, 0],
                radius=0.1,
                color=MATERIAL_MINT,
                stroke_color=MATERIAL_MINT_STROKE,
                stroke_width=3
            )
            center_dots.add(dot)

        self.play(FadeIn(center_dots), run_time=2)
        self.wait(2)

        # Step 3: Show compression
        step3 = MonospaceText(
            f"Step 3: Compressed {len(sampled_values)} → {n_clusters} values",
            font_size=24,
            color=get_text_color()
        )
        self.play(Transform(step1, step3))
        self.wait(2)

        # Fade out
        self.play(
            FadeOut(scatter_dots),
            FadeOut(center_dots),
            FadeOut(step1),
            FadeOut(title)
        )


if __name__ == "__main__":
    # Test rendering
    import os
    os.system("manim -pql layer_clustering_viz.py LayerClusteringVisualization")
