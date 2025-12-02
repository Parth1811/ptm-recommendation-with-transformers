from manim import *


class NeuralNetwork(VGroup):
    """Neural network visualization with configurable layers."""

    def __init__(
        self,
        layers,
        layer_spacing=2,
        node_radius=0.15,
        node_opacity=0.8,
        node_color=BLUE,
        edge_color=WHITE,
        edge_opacity=0.6,
        show_labels=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers = layers
        self.layer_spacing = layer_spacing
        self.node_radius = node_radius
        self.node_opacity = node_opacity

        # Create layer groups
        self.layer_groups = VGroup()
        self.edges = VGroup()

        # Create nodes for each layer
        for layer_idx, num_nodes in enumerate(layers):
            layer_group = VGroup()
            node_spacing = 0.6

            for node_idx in range(num_nodes):
                # Center the layer vertically
                y_offset = (num_nodes - 1) * node_spacing / 2
                node = Circle(
                    radius=node_radius,
                    fill_color=node_color,
                    fill_opacity=self.node_opacity,
                    stroke_color=WHITE,
                    stroke_width=2,
                )
                node.move_to(
                    RIGHT * layer_idx * layer_spacing
                    + UP * (node_idx * node_spacing - y_offset)
                )
                layer_group.add(node)

            self.layer_groups.add(layer_group)

        # Create edges between layers
        for i in range(len(layers) - 1):
            current_layer = self.layer_groups[i]
            next_layer = self.layer_groups[i + 1]

            for node1 in current_layer:
                for node2 in next_layer:
                    edge = Line(
                        node1.get_center(),
                        node2.get_center(),
                        stroke_color=edge_color,
                        stroke_opacity=edge_opacity,
                        stroke_width=2,
                    )
                    self.edges.add(edge)

        # Add edges first (so they appear behind nodes)
        self.add(self.edges)
        self.add(self.layer_groups)

        # Add layer labels if requested
        if show_labels:
            self.labels = VGroup()
            layer_names = ["Input"] + [f"Hidden {i}" for i in range(1, len(layers) - 1)] + ["Output"]

            for idx, (layer_group, name) in enumerate(zip(self.layer_groups, layer_names)):
                label = Text(name, font_size=20)
                label.next_to(layer_group, DOWN, buff=0.3)
                self.labels.add(label)

            self.add(self.labels)

        # Center the entire network
        self.move_to(ORIGIN)

    def animate_forward_pass(self, scene, run_time=2):
        """Animate data flowing through the network."""
        animations = []

        # Highlight layers sequentially
        for layer in self.layer_groups:
            animations.append(
                layer.animate.set_fill(YELLOW, opacity=1)
            )
            scene.play(*animations, run_time=run_time / len(self.layers))
            animations = []
            scene.play(layer.animate.set_fill(BLUE, opacity=self.node_opacity), run_time=0.2)
