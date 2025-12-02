from color_constants import (MATERIAL_BLUE, MATERIAL_GREEN, MATERIAL_RED,
                             ColorTheme, get_text_color)
from dataset_pipeline import DatasetPipeline
from manim import *
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import DatasetTokens, ModelTokens, Tokens
from transformer import CrossAttentionBlock, Transformer


class ShowcaseAll(Scene):
    """Showcase all animation components one by one."""

    def construct(self):
        # Title
        # title = Text("ML Pipeline Components", font_size=48)
        # self.play(Write(title), run_time=2)
        # self.play(FadeOut(title))

        # 1. Neural Network
        # self.show_neural_network()

        # # 2. Tokens
        # self.show_tokens()

        # # 3. Cross Attention Block
        # self.show_cross_attention()

        # # 4. Transformer
        # self.show_transformer()

        # # 5. Dataset Pipeline
        # self.show_dataset_pipeline()

        # # 6. Round Box
        # self.show_round_box()

        # End
        end_text = Text("End of Showcase", font_size=36, color=get_text_color())
        self.play(Write(end_text), run_time=2)
        self.wait(1)

    def show_neural_network(self):
        """Display neural network."""
        label = Text("Neural Network", font_size=32, color=get_text_color())
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
        label = Text("Token Visualizations", font_size=32, color=get_text_color())
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
        label = Text("Cross Attention Block", font_size=32, color=get_text_color())
        label.to_edge(UP)

        attention = CrossAttentionBlock()

        self.play(Write(label))
        self.play(Create(attention))
        self.wait(2)
        self.play(FadeOut(attention), FadeOut(label))

    def show_transformer(self):
        """Display transformer architecture."""
        label = Text("Transformer Architecture", font_size=32, color=get_text_color())
        label.to_edge(UP)

        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.7)

        self.play(Write(label))
        self.play(Create(transformer))
        self.wait(2)
        self.play(FadeOut(transformer), FadeOut(label))

    def show_dataset_pipeline(self):
        """Display dataset processing pipeline."""
        label = Text("Dataset Processing Pipeline", font_size=32, color=get_text_color())
        label.to_edge(UP)

        pipeline = DatasetPipeline()
        pipeline.scale(0.7)
        pipeline.shift(DOWN * 0.3)

        self.play(Write(label))
        self.play(Create(pipeline), run_time=3)
        self.wait(1)

        # Animate the forward pass (extended time for 3 samples)
        pipeline.animate_forward(self, run_time=12)

        self.wait(2)
        self.play(FadeOut(pipeline), FadeOut(label))

    def show_round_box(self):
        """Display round box utility."""
        label = Text("Round Box Utility", font_size=32, color=get_text_color())
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


class RecommendationPipeline(Scene):
    """Show the complete recommendation pipeline with model pipeline, dataset pipeline, and transformer."""

    def construct(self):
        # Title
        title = Text("Model Recommendation Pipeline", font_size=40, color=get_text_color())
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # ===== MODEL PIPELINE (Top Left) =====
        model_network = NeuralNetwork(
            layers=[4, 3, 3, 4],
            node_radius=0.15,
            node_opacity=1,
            layer_spacing=0.8,
        )
        model_network.scale(0.8)
        model_network.shift(LEFT * 5 + UP * 1.5)

        model_pipeline_label = RoundBox(
            "Model\nPipeline",
            width=1.5,
            height=1.2,
            fill_color=MATERIAL_BLUE,
            text_align="center",
            font_size=18,
        )
        model_pipeline_label.next_to(model_network, RIGHT, buff=0.3)

        model_tokens = ModelTokens(num_models=3, abbreviated=True)
        model_tokens.scale(0.8)
        model_tokens.next_to(model_pipeline_label, RIGHT, buff=0.3)

        # ===== DATASET PIPELINE (Bottom Left) =====
        dataset_pipeline = DatasetPipeline()
        dataset_pipeline.next_to(model_network, DOWN, buff=0.5)
        dataset_pipeline.scale(0.6)
        dataset_pipeline.shift(RIGHT * 4 + UP * 0.5)


        # ===== TRANSFORMER (Right) =====
        transformer = Transformer(show_fc_layer=True)
        transformer.next_to(model_tokens, RIGHT)
        transformer.scale(0.6)
        transformer.shift(DOWN * 1.5 + LEFT * 1)

        # Create all components
        self.play(Create(model_network), run_time=2)
        self.play(Create(model_pipeline_label), run_time=1)
        self.play(Create(model_tokens), run_time=1)

        self.play(Create(dataset_pipeline), run_time=2)

        self.play(Create(transformer), run_time=2)
        self.wait(1)

        # Animate model pipeline forward pass
        model_network.animate_forward_pass(self, run_time=3)
        self.wait(1)

        # Animate dataset pipeline forward pass
        dataset_pipeline.animate_forward(self, run_time=8)
        self.wait(1)

        # Fade out all components
        self.play(
            FadeOut(title),
            FadeOut(model_network),
            FadeOut(model_pipeline_label),
            FadeOut(model_tokens),
            FadeOut(dataset_pipeline),
            FadeOut(transformer),
        )
        self.wait(1)

