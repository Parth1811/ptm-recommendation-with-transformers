"""Birds-eye view of transformer attention mechanisms (cross-attention and self-attention)."""
from color_constants import (get_arrow_color, get_attention_block_color,
                             get_text_color)
from manim import *
from tokens import DatasetTokens, ModelTokens
from transformer import Transformer


class CrossAttentionView(VGroup):
    """Cross-attention layout: Model and dataset tokens stacked vertically."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model tokens on top left
        self.model_tokens = ModelTokens(num_models=2, abbreviated=True)
        self.model_tokens.scale(0.9)
        self.model_tokens.move_to(UP * 1.5 + LEFT * 2.5)

        # Dataset tokens below
        self.dataset_tokens = DatasetTokens(num_samples=2, abbreviated=True)
        self.dataset_tokens.next_to(self.model_tokens, DOWN, buff=1.0)
        self.dataset_tokens.scale(0.9)
        # self.dataset_tokens.move_to(DOWN * 1.5 + LEFT * 2.5)

        # Transformer on the right
        self.transformer = Transformer(show_fc_layer=True)
        self.transformer.scale(0.7)
        self.transformer.move_to(RIGHT * 3)

        # Two arrows from tokens to transformer
        self.model_arrow = Arrow(
            self.model_tokens.get_right() + RIGHT * 0.1,
            self.transformer.get_left() + UP * 0.5,
            color=get_arrow_color(),
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25,
            buff=0
        )
        self.dataset_arrow = Arrow(
            self.dataset_tokens.get_right() + RIGHT * 0.1,
            self.transformer.get_left() + DOWN * 0.5,
            color=get_arrow_color(),
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25,
            buff=0
        )

        self.add(self.model_tokens, self.dataset_tokens, self.transformer)
        self.add(self.model_arrow, self.dataset_arrow)


class SelfAttentionView(VGroup):
    """Self-attention layout: Model and dataset tokens side by side horizontally."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model tokens on the left
        self.model_tokens = ModelTokens(num_models=2, abbreviated=True)
        self.model_tokens.scale(0.9)
        self.model_tokens.move_to(LEFT * 4)

        # Dataset tokens next to model tokens
        self.dataset_tokens = DatasetTokens(num_samples=2, abbreviated=True)
        self.dataset_tokens.next_to(self.model_tokens, RIGHT, buff=0.5)
        self.dataset_tokens.scale(0.9)
        # self.dataset_tokens.move_to(LEFT * 0.8)

        # Transformer on the right
        self.transformer = Transformer(show_fc_layer=True)
        self.transformer.next_to(self.dataset_tokens, RIGHT, buff=1.0)
        self.transformer.scale(0.7)
        # self.transformer.move_to(RIGHT * 2.5)

        # Single arrow from dataset tokens (representing combined input)
        self.dataset_arrow = Arrow(
            self.dataset_tokens.get_right() + RIGHT * 0.1,
            self.transformer.get_left() + LEFT * 0.1,
            color=get_arrow_color(),
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25,
            buff=0
        )

        self.add(self.model_tokens, self.dataset_tokens, self.transformer)
        self.add(self.dataset_arrow)


class BirdsEyeView(VGroup):
    """Birds-eye view of transformer with switchable attention modes."""

    def __init__(self, mode="cross", **kwargs):
        super().__init__(**kwargs)

        self.mode = mode

        # Create both views
        self.cross_view = CrossAttentionView()
        self.self_view = SelfAttentionView()

        # Add the initial view based on mode
        if mode == "cross":
            # self.add(self.cross_view)
            self.current_view = self.cross_view.copy()
        else:
            # self.add(self.self_view)
            self.current_view = self.self_view.copy()

        self.add(self.current_view)
        self.move_to(ORIGIN)

    def switch_to_cross_attention(self, scene, duration=1.0):
        """Animate transition from self-attention to cross-attention."""
        if self.mode == "cross":
            return  # Already in cross-attention mode

        # Create a new model arrow for the current view (don't use cross_view's arrow)
        new_model_arrow = Arrow(
            self.cross_view.model_tokens.get_right() + RIGHT * 0.1,
            self.cross_view.transformer.get_left() + UP * 0.5,
            color=get_arrow_color(),
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25,
            buff=0
        )

        # Transform from self view to cross view
        scene.play(
            Transform(self.current_view.model_tokens, self.cross_view.model_tokens),
            Transform(self.current_view.dataset_tokens, self.cross_view.dataset_tokens),
            Transform(self.current_view.transformer, self.cross_view.transformer),
            Transform(self.current_view.dataset_arrow, self.cross_view.dataset_arrow),
            FadeIn(new_model_arrow),
            run_time=duration
        )

        # Add the new model arrow to current_view
        self.current_view.model_arrow = new_model_arrow
        self.current_view.add(new_model_arrow)

        self.mode = "cross"

    def switch_to_self_attention(self, scene, duration=1.0):
        """Animate transition from cross-attention to self-attention."""
        if self.mode == "self":
            return  # Already in self-attention mode

        # Transform from cross view to self view
        scene.play(
            Transform(self.current_view.model_tokens, self.self_view.model_tokens),
            Transform(self.current_view.dataset_tokens, self.self_view.dataset_tokens),
            Transform(self.current_view.transformer, self.self_view.transformer),
            Transform(self.current_view.dataset_arrow, self.self_view.dataset_arrow),
            FadeOut(self.current_view.model_arrow),
            run_time=duration
        )

        # Remove the model arrow from current_view
        self.current_view.remove(self.current_view.model_arrow)
        self.current_view.model_arrow = None

        self.mode = "self"

    def toggle_attention_mode(self, scene, duration=1.0):
        """Toggle between cross-attention and self-attention modes."""
        if self.mode == "cross":
            self.switch_to_self_attention(scene, duration)
        else:
            self.switch_to_cross_attention(scene, duration)
