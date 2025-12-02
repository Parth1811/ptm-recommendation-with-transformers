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
        title = Text("ML Pipeline Components", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # 1. Neural Network
        self.show_neural_network()

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
        end_text = Text("End of Showcase", font_size=36)
        self.play(Write(end_text))
        self.wait(2)

    def show_neural_network(self):
        """Display neural network."""
        label = Text("Neural Network", font_size=32)
        label.to_edge(UP)

        nn = NeuralNetwork(
            layers=[3, 5, 4, 2],
            node_radius=0.2,
            node_opacity=1,
            layer_spacing=1
        )
        nn.scale(1.6)

        self.play(Write(label))
        self.play(Create(nn))
        nn.animate_forward_pass(self, run_time=2)
        self.wait(1)
        self.play(FadeOut(nn), FadeOut(label))

    def show_tokens(self):
        """Display different token types."""
        label = Text("Token Visualizations", font_size=32)
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
        label = Text("Cross Attention Block", font_size=32)
        label.to_edge(UP)

        attention = CrossAttentionBlock()

        self.play(Write(label))
        self.play(Create(attention))
        self.wait(2)
        self.play(FadeOut(attention), FadeOut(label))

    def show_transformer(self):
        """Display transformer architecture."""
        label = Text("Transformer Architecture", font_size=32)
        label.to_edge(UP)

        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.7)

        self.play(Write(label))
        self.play(Create(transformer))
        self.wait(2)
        self.play(FadeOut(transformer), FadeOut(label))

    def show_dataset_pipeline(self):
        """Display dataset processing pipeline."""
        label = Text("Dataset Processing Pipeline", font_size=32)
        label.to_edge(UP)

        pipeline = DatasetPipeline()
        pipeline.scale(0.5).shift(DOWN * 0.5)

        self.play(Write(label))
        self.play(Create(pipeline), run_time=3)
        self.wait(2)
        self.play(FadeOut(pipeline), FadeOut(label))

    def show_round_box(self):
        """Display round box utility."""
        label = Text("Round Box Utility", font_size=32)
        label.to_edge(UP)

        boxes = VGroup()

        box1 = RoundBox("Example 1", width=2, height=1, fill_color=BLUE)
        box2 = RoundBox("Example 2", width=2.5, height=1.2, fill_color=RED)
        box3 = RoundBox("Example 3", width=1.8, height=0.8, fill_color=GREEN)

        box1.shift(LEFT * 3)
        box3.shift(RIGHT * 3)

        boxes.add(box1, box2, box3)

        self.play(Write(label))
        self.play(Create(boxes))
        self.wait(2)
        self.play(FadeOut(boxes), FadeOut(label))


class NeuralNetworkScene(Scene):
    """Dedicated scene for neural network."""

    def construct(self):
        nn = NeuralNetwork(layers=[4, 6, 6, 3])
        self.play(Create(nn))
        nn.animate_forward_pass(self, run_time=3)
        self.wait(2)


class TransformerScene(Scene):
    """Dedicated scene for transformer."""

    def construct(self):
        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.8)
        self.play(Create(transformer))
        self.wait(3)


class DatasetPipelineScene(Scene):
    """Dedicated scene for dataset pipeline."""

    def construct(self):
        pipeline = DatasetPipeline()
        pipeline.scale(0.6)
        self.play(Create(pipeline), run_time=3)
        self.wait(3)


class TokensScene(Scene):
    """Dedicated scene for tokens."""

    def construct(self):
        model_tokens = ModelTokens(num_models=5)
        model_tokens.shift(UP * 2)

        dataset_tokens = DatasetTokens(num_samples=3)
        dataset_tokens.shift(DOWN * 2)

        self.play(Create(model_tokens))
        self.wait(1)
        self.play(Create(dataset_tokens))
        self.wait(2)
