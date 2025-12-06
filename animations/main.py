from birdseyeview import BirdsEyeView
from color_constants import (MATERIAL_BLUE, MATERIAL_GREEN, MATERIAL_RED,
                             ColorTheme, get_text_color)
from dataset_pipeline import DatasetPipeline
from manim import *
from model_pipeline import ModelPipeline
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import DatasetTokens, ModelTokens, Tokens
from transformer import AttentionBlock, Transformer


class ShowcaseAll(Scene):
    """Showcase all animation components one by one."""

    def construct(self):
        # Title
        # title = MonospaceText("ML Pipeline Components", font_size=48)
        # self.play(Write(title), run_time=2)
        # self.play(FadeOut(title))

        # 1. Neural Network
        # self.show_neural_network()

        # # 2. Tokens
        # self.show_tokens()

        # # 3. Cross Attention Block
        # self.show_cross_attention()

        # # 4. Transformer
        self.show_transformer()

        # 5. Dataset Pipeline
        # self.show_dataset_pipeline()

        # # 6. Round Box
        # self.show_round_box()

        # # 7. Birds-Eye View
        # self.show_birds_eye_view()

        # 8. Model Pipeline
        # self.show_model_pipeline()

        # End
        end_text = MonospaceText("End of Showcase", font_size=36, color=get_text_color())
        self.play(Write(end_text), run_time=2)
        self.wait(1)

    def show_neural_network(self):
        """Display neural network."""
        label = MonospaceText("Neural Network", font_size=32, color=get_text_color())
        label.to_edge(UP)

        nn = NeuralNetwork(
            layers=[3, 5, 4, 2],
            node_radius=0.2,
            node_opacity=1,
            layer_spacing=1,
            abbreviated_hidden=True,
            abbreviated_spacing=0.8,
        )
        nn.scale(1.6)

        self.play(Write(label))
        self.play(Create(nn))
        nn.animate_forward_pass(self, run_time=5)
        self.wait(1)
        self.play(FadeOut(nn), FadeOut(label))

    def show_tokens(self):
        """Display different token types."""
        label = MonospaceText("Token Visualizations", font_size=32, color=get_text_color())
        label.to_edge(UP)

        # Model tokens
        model_tokens = ModelTokens(num_models=3)
        model_tokens.shift(LEFT * 3)

        # Dataset tokens
        dataset_tokens = DatasetTokens(num_samples=4)
        dataset_tokens.shift(RIGHT * 3)

        self.play(Write(label))
        self.play(Create(model_tokens), Create(dataset_tokens))
        self.wait(2)
        self.play(FadeOut(model_tokens), FadeOut(dataset_tokens), FadeOut(label))

    def show_cross_attention(self):
        """Display cross-attention block."""
        label = MonospaceText("Cross Attention Block", font_size=32, color=get_text_color())
        label.to_edge(UP)

        attention = AttentionBlock()

        self.play(Write(label))
        self.play(Create(attention))
        self.wait(2)
        self.play(FadeOut(attention), FadeOut(label))

    def show_transformer(self):
        """Display transformer architecture."""
        label = MonospaceText("Transformer Architecture", font_size=32, color=get_text_color())
        label.to_edge(UP)

        transformer = Transformer(
            show_fc_layer=True,
            show_probability_distribution=False,
            show_pdf=True,
            mode="cross"
        )
        transformer.scale(0.7)

        self.play(Write(label))
        self.play(Create(transformer))
        self.wait(2)

        transformer.animate_show_pdf(self, duration=2)

        self.play(FadeOut(transformer), FadeOut(label))

    def show_dataset_pipeline(self):
        """Display dataset processing pipeline."""
        label = MonospaceText("Dataset Processing Pipeline", font_size=32, color=get_text_color())
        label.to_edge(UP)

        pipeline = DatasetPipeline()
        pipeline.scale(0.7)
        pipeline.shift(DOWN * 0.3)

        self.play(Write(label))
        self.play(Create(pipeline), run_time=3)
        self.wait(1)

        # Animate the forward pass (extended time for 3 samples)
        pipeline.animate_forward(self, run_time=8)

        self.wait(2)
        self.play(FadeOut(pipeline), FadeOut(label))

    def show_round_box(self):
        """Display round box utility."""
        label = MonospaceText("Round Box Utility", font_size=32, color=get_text_color())
        label.to_edge(UP)

        boxes = VGroup()

        box1 = RoundBox("Example 1", width=2, height=1, fill_color=MATERIAL_BLUE)
        box2 = RoundBox("Example 2", width=2.5, height=1.2, fill_color=MATERIAL_RED)
        box3 = RoundBox("Example 3", width=1.8, height=0.8, fill_color=MATERIAL_GREEN)

        box1.shift(LEFT * 3)
        box3.shift(RIGHT * 3)

        boxes.add(box1, box2, box3)

        self.play(Write(label))
        self.play(Create(boxes))
        self.wait(2)
        self.play(FadeOut(boxes), FadeOut(label))

    def show_model_pipeline(self):
        """Display model pipeline utility."""
        label = MonospaceText("Model Pipeline Utility", font_size=32, color=get_text_color())
        label.to_edge(UP)

        model_pipeline = ModelPipeline()
        model_pipeline.scale(0.8)

        self.play(Write(label))
        self.play(Create(model_pipeline), run_time=3)
        self.wait(1)

        # Animate the forward pass
        model_pipeline.animate_forward(self, run_time=12)

        self.wait(2)
        self.play(FadeOut(model_pipeline), FadeOut(label))

    def show_birds_eye_view(self):
        """Display birds-eye view with attention mechanism switching."""
        label = MonospaceText("Birds-Eye View: Attention Mechanisms", font_size=32, color=get_text_color())
        label.to_edge(UP)

        # Start with cross-attention
        birdseyeview = BirdsEyeView(mode="self")
        birdseyeview.scale(0.8)

        self.play(Write(label))
        self.play(Create(birdseyeview))
        self.wait(2)

        # Animate transition to self-attention
        birdseyeview.toggle_attention_mode(self, duration=2.0)
        self.wait(2)

        # Animate back to cross-attention
        birdseyeview.toggle_attention_mode(self, duration=2.0)
        self.wait(2)

        self.play(FadeOut(birdseyeview), FadeOut(label))
