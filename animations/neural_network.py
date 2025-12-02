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
        edge_color=BLUE_E,
        edge_opacity=0.6,
        show_labels=True,
        abbreviated_hidden=False,
        abbreviated_spacing=0.8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers = layers
        self.layer_spacing = layer_spacing
        self.node_radius = node_radius
        self.node_opacity = node_opacity
        self.node_color = node_color
        self.edge_color = edge_color
        self.edge_opacity = edge_opacity
        self.abbreviated_hidden = abbreviated_hidden
        # Cap abbreviated_spacing to layer_spacing
        self.abbreviated_spacing = min(abbreviated_spacing, layer_spacing)

        # Create layer groups
        self.layer_groups = VGroup()
        self.edges = VGroup()
        self.ellipsis = None

        # Determine which layers to show
        if abbreviated_hidden and len(layers) >= 4:
            # Show first 2 layers, ellipsis, then last 2 layers
            layers_to_show = [0, 1, "...", len(layers) - 2, len(layers) - 1]
            visual_positions = [0, 1, 2, 3, 4]
        else:
            layers_to_show = list(range(len(layers)))
            visual_positions = list(range(len(layers)))

        # Create nodes for each layer
        # Track visual positions for edge creation
        self.layer_visual_positions = []

        for visual_idx, layer_spec in enumerate(layers_to_show):
            if layer_spec == "...":
                # Create ellipsis indicator
                self.ellipsis = Text("...", font_size=48, color=WHITE)
                self.ellipsis.move_to(RIGHT * visual_positions[visual_idx] * layer_spacing)
                continue

            layer_idx = layer_spec
            num_nodes = layers[layer_idx]
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
                    RIGHT * visual_positions[visual_idx] * layer_spacing
                    + UP * (node_idx * node_spacing - y_offset)
                )
                layer_group.add(node)

            self.layer_groups.add(layer_group)
            self.layer_visual_positions.append(visual_positions[visual_idx])

        # Create edges between layers (organized by layer pair for animation)
        self.edge_groups = VGroup()
        for i in range(len(self.layer_groups) - 1):
            current_layer = self.layer_groups[i]
            next_layer = self.layer_groups[i + 1]

            # Check if there's a gap (ellipsis) between these layers
            current_pos = self.layer_visual_positions[i]
            next_pos = self.layer_visual_positions[i + 1]

            layer_edges = VGroup()

            # If there's a gap (ellipsis), create half-edges
            if next_pos - current_pos > 1:
                # Calculate the total horizontal distance between layers
                total_horizontal_distance = (next_pos - current_pos) * layer_spacing

                # Calculate edge fraction: how far along the direction vector each edge extends
                # This leaves abbreviated_spacing gap in the middle
                edge_fraction = (total_horizontal_distance - self.abbreviated_spacing) / (2 * total_horizontal_distance)

                # Ensure edge_fraction is positive (in case abbreviated_spacing is very large)
                edge_fraction = max(0.05, edge_fraction)

                # Create half-edges going OUT from current layer towards ellipsis
                # Fan out to all nodes in next layer for proper connectivity visualization
                for node1 in current_layer:
                    for node2 in next_layer:
                        node1_center = node1.get_center()
                        node2_center = node2.get_center()

                        # Calculate the direction vector from node1 to node2
                        direction = node2_center - node1_center
                        # Create edge extending by edge_fraction
                        end_point = node1_center + direction * edge_fraction

                        edge = Line(
                            node1_center,
                            end_point,
                            stroke_color=edge_color,
                            stroke_opacity=edge_opacity,
                            stroke_width=2,
                        )
                        layer_edges.add(edge)
                        self.edges.add(edge)

                # Create half-edges coming INTO next layer from ellipsis
                for node1 in current_layer:
                    for node2 in next_layer:
                        node1_center = node1.get_center()
                        node2_center = node2.get_center()

                        # Calculate the direction vector from node1 to node2
                        direction = node2_center - node1_center
                        # Create edge starting from (1 - edge_fraction) point
                        start_point = node1_center + direction * (1 - edge_fraction)

                        edge = Line(
                            start_point,
                            node2_center,
                            stroke_color=edge_color,
                            stroke_opacity=edge_opacity,
                            stroke_width=2,
                        )
                        layer_edges.add(edge)
                        self.edges.add(edge)
            else:
                # Normal full edges for consecutive layers
                for node1 in current_layer:
                    for node2 in next_layer:
                        edge = Line(
                            node1.get_center(),
                            node2.get_center(),
                            stroke_color=edge_color,
                            stroke_opacity=edge_opacity,
                            stroke_width=2,
                        )
                        layer_edges.add(edge)
                        self.edges.add(edge)

            self.edge_groups.add(layer_edges)

        # Add edges first (so they appear behind nodes)
        self.add(self.edges)
        self.add(self.layer_groups)

        # Add ellipsis if present
        if self.ellipsis:
            self.add(self.ellipsis)

        # Add layer labels if requested
        if show_labels:
            self.labels = VGroup()

            if abbreviated_hidden and len(layers) > 4:
                # Create labels for visible layers only
                layer_names = [
                    "Input",
                    "Hidden 1",
                    "Hidden ...",
                    f"Hidden {len(layers) - 2}",
                    "Output"
                ]

                # Add labels for actual layer groups (skip ellipsis position)
                label_idx = 0
                for visual_idx, layer_spec in enumerate(layers_to_show):
                    if layer_spec == "...":
                        # Add label for ellipsis
                        label = Text(layer_names[visual_idx], font_size=20)
                        label.next_to(self.ellipsis, DOWN, buff=0.3)
                        self.labels.add(label)
                    else:
                        label = Text(layer_names[visual_idx], font_size=20)
                        label.next_to(self.layer_groups[label_idx], DOWN, buff=0.3)
                        self.labels.add(label)
                        label_idx += 1
            else:
                layer_names = ["Input"] + [f"Hidden {i}" for i in range(1, len(layers) - 1)] + ["Output"]
                for idx, (layer_group, name) in enumerate(zip(self.layer_groups, layer_names)):
                    label = Text(name, font_size=20)
                    label.next_to(layer_group, DOWN, buff=0.3)
                    self.labels.add(label)

            self.add(self.labels)

        # Center the entire network
        self.move_to(ORIGIN)

    def animate_forward_pass(self, scene, run_time=2, flow_color=YELLOW):
        """Animate data flowing through the network with smooth transitions and wiggling edges."""
        layer_time = run_time / len(self.layer_groups)

        for i, layer in enumerate(self.layer_groups):
            # Activate current layer nodes
            scene.play(
                layer.animate.set_fill(flow_color, opacity=1),
                run_time=layer_time * 0.4,
                rate_func=smooth
            )

            # If not the last layer, animate edges to next layer
            if i < len(self.edge_groups) and len(self.edge_groups[i]) > 0:
                # Create wiggle animations for each edge in the group
                edge_animations = []
                for eidx, edge in enumerate(self.edge_groups[i]):
                    edge_animations.append(
                        Wiggle(
                            edge,
                            scale_value=1.05,
                            n_wiggles=3,
                            rotation_angle=(1 if eidx % 2 == 0 else -1) * 0.015 * TAU
                        )
                    )

                # Light up edges flowing to next layer with wiggle effect
                scene.play(
                    self.edge_groups[i].animate.set_stroke(flow_color, opacity=1, width=3),
                    *edge_animations,
                    run_time=layer_time * 0.5,
                    rate_func=smoothstep
                )

            # Deactivate current layer (fade back)
            deactivate_anims = [
                layer.animate.set_fill(self.node_color, opacity=self.node_opacity)
            ]

            # Also fade edges back if they were lit
            if i < len(self.edge_groups) and len(self.edge_groups[i]) > 0:
                deactivate_anims.append(
                    self.edge_groups[i].animate.set_stroke(self.edge_color, opacity=self.edge_opacity, width=2)
                )

            scene.play(
                *deactivate_anims,
                run_time=layer_time * 0.3,
                rate_func=smooth
            )
