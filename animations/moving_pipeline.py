"""Moving camera scene showing the complete recommendation pipeline with dynamic camera movements."""
from color_constants import (MATERIAL_BLUE, MATERIAL_GREEN, MATERIAL_RED,
                             get_arrow_color, get_text_color)
from dataset_pipeline import DatasetPipeline
from manim import *
from model_pipeline import ModelPipeline
from monospace_text import MonospaceText
from neural_network import NeuralNetwork
from round_box import RoundBox
from tokens import DatasetTokens, ModelTokens
from transformer import Transformer


class MovingRecommendationPipeline(MovingCameraScene):
    """
    Complete recommendation pipeline with MovingCameraScene for dynamic panning.

    Shows:
    1. Bird's eye view of the entire pipeline
    2. Camera pans to model token pipeline (top)
    3. Camera pans to dataset pipeline (bottom)
    4. Returns to bird's eye view showing transformer integration
    """

    def construct(self):
        # ===== SETUP ALL COMPONENTS (positioned for wide layout) =====

        # Title (stays at top during bird's eye view)
        title = MonospaceText("Model Recommendation Pipeline", font_size=40, color=get_text_color())
        title.to_edge(UP)

        # ===== MODEL PIPELINE (Top section) =====
        # Position model pipeline on the left-top area
        model_network = NeuralNetwork(
            layers=[4, 3, 3, 4],
            node_radius=0.15,
            node_opacity=1,
            layer_spacing=0.8,
        )
        model_network.scale(0.8)
        model_network.move_to(LEFT * 5.2 + UP * 1.5)

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

        # Arrow from model network to label
        model_arrow1 = Arrow(
            model_network.get_right(),
            model_pipeline_label.get_left(),
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        # Arrow from label to tokens
        model_arrow2 = Arrow(
            model_pipeline_label.get_right(),
            model_tokens.get_left(),
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        # ===== DATASET PIPELINE (Bottom section) =====
        # Position dataset pipeline on the left-bottom area
        dataset_pipeline = DatasetPipeline()
        dataset_pipeline.next_to(model_network, DOWN, buff=0.5)
        dataset_pipeline.move_to(LEFT * 4 + DOWN * 2.5)
        dataset_pipeline.scale(0.6)


        # Get dataset tokens from the pipeline for connection
        dataset_tokens = dataset_pipeline.dataset_tokens

        # ===== TRANSFORMER (Right center) =====
        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.6)
        transformer.move_to(RIGHT * 5 + UP * 0)

        # ===== ARROWS TO TRANSFORMER =====
        # Arrow from model tokens to transformer (Q - Query)
        arrow_q = Arrow(
            model_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + UP * 1.0,
            buff=0.2,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
            color=MATERIAL_GREEN,
        )
        q_label = MonospaceText("Q (Query)", font_size=16, color=MATERIAL_GREEN)
        q_label.next_to(arrow_q.get_center(), UP, buff=0.1)

        # Arrow from dataset tokens to transformer (K, V - Keys, Values)
        arrow_kv = Arrow(
            dataset_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + DOWN * 1.0,
            buff=0.2,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
            color=MATERIAL_RED,
        )
        kv_label = MonospaceText("K, V (Keys, Values)", font_size=16, color=MATERIAL_RED)
        kv_label.next_to(arrow_kv.get_center(), DOWN, buff=0.1)

        # ===== OUTPUT =====
        output_box = RoundBox(
            "Model\nRankings",
            width=2.0,
            height=1.5,
            fill_color=MATERIAL_GREEN,
            text_align="center",
            font_size=20,
        )
        output_box.next_to(transformer, RIGHT, buff=1.0)

        output_arrow = Arrow(
            transformer.get_right(),
            output_box.get_left(),
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
            color=get_arrow_color(),
        )

        # ===== ANIMATION SEQUENCE =====

        # 1. BIRD'S EYE VIEW - Show all components at once
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)

        # Add all components to the scene
        self.play(
            Create(model_network),
            Create(model_pipeline_label),
            Create(model_tokens),
            Create(model_arrow1),
            Create(model_arrow2),
            run_time=2,
        )

        self.play(
            Create(dataset_pipeline),
            run_time=2,
        )

        self.play(
            Create(transformer),
            Create(arrow_q),
            Create(q_label),
            Create(arrow_kv),
            Create(kv_label),
            run_time=2,
        )

        self.play(
            Create(output_box),
            Create(output_arrow),
            run_time=1.5,
        )

        self.wait(2)

        # 2. PAN TO MODEL PIPELINE (Top)
        # Calculate the target position for camera to focus on model pipeline
        model_pipeline_center = VGroup(
            model_network, model_pipeline_label, model_tokens, model_arrow1, model_arrow2
        ).get_center()

        # Add a section title for model pipeline
        model_section_title = MonospaceText("Model Embedding Pipeline", font_size=36, color=get_text_color())
        model_section_title.move_to(model_pipeline_center + UP * 2)

        # Pan camera to model pipeline and zoom in
        self.play(
            self.camera.frame.animate.move_to(model_pipeline_center).set(width=12),
            FadeIn(model_section_title),
            run_time=2,
        )
        self.wait(1)

        # Animate model network forward pass
        model_network.animate_forward_pass(self, run_time=3)
        self.wait(1.5)

        # Highlight the model tokens
        self.play(
            model_tokens.animate.scale(1.2),
            run_time=0.5,
        )
        self.play(
            model_tokens.animate.scale(1/1.2),
            run_time=0.5,
        )
        self.wait(1)

        # Fade out section title
        self.play(FadeOut(model_section_title), run_time=1)

        # 3. PAN TO DATASET PIPELINE (Bottom)
        # Calculate the target position for camera to focus on dataset pipeline
        dataset_pipeline_center = dataset_pipeline.get_center()

        # Add a section title for dataset pipeline
        dataset_section_title = MonospaceText("Dataset Embedding Pipeline", font_size=36, color=get_text_color())
        dataset_section_title.move_to(dataset_pipeline_center + UP * 3.5)

        # Pan camera to dataset pipeline
        self.play(
            self.camera.frame.animate.move_to(dataset_pipeline_center + UP * 0.5).set(width=16),
            FadeIn(dataset_section_title),
            run_time=2,
        )
        self.wait(1)

        # Animate dataset pipeline forward pass
        dataset_pipeline.animate_forward(self, run_time=10)
        self.wait(1.5)

        # Fade out section title
        self.play(FadeOut(dataset_section_title), run_time=1)

        # 4. RETURN TO BIRD'S EYE VIEW - Show transformer integration
        # Pan back to full view
        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=18),
            run_time=2,
        )
        self.wait(1)

        # Highlight the flow to transformer
        # Pulse the Q arrow and label
        self.play(
            arrow_q.animate.set_stroke(width=4),
            q_label.animate.scale(1.3),
            run_time=0.8,
        )
        self.play(
            arrow_q.animate.set_stroke(width=2),
            q_label.animate.scale(1/1.3),
            run_time=0.8,
        )

        # Pulse the KV arrow and label
        self.play(
            arrow_kv.animate.set_stroke(width=4),
            kv_label.animate.scale(1.3),
            run_time=0.8,
        )
        self.play(
            arrow_kv.animate.set_stroke(width=2),
            kv_label.animate.scale(1/1.3),
            run_time=0.8,
        )

        # Highlight transformer
        self.play(
            transformer.animate.scale(1.15),
            run_time=0.8,
        )
        self.play(
            transformer.animate.scale(1/1.15),
            run_time=0.8,
        )

        # Highlight output
        self.play(
            output_arrow.animate.set_stroke(width=4),
            output_box.animate.scale(1.2),
            run_time=0.8,
        )
        self.play(
            output_arrow.animate.set_stroke(width=2),
            output_box.animate.scale(1/1.2),
            run_time=0.8,
        )

        self.wait(2)

        # 5. FINAL ZOOM OUT
        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=20),
            run_time=2,
        )
        self.wait(2)

        # Fade out everything
        self.play(
            FadeOut(title),
            FadeOut(model_network),
            FadeOut(model_pipeline_label),
            FadeOut(model_tokens),
            FadeOut(model_arrow1),
            FadeOut(model_arrow2),
            FadeOut(dataset_pipeline),
            FadeOut(transformer),
            FadeOut(arrow_q),
            FadeOut(q_label),
            FadeOut(arrow_kv),
            FadeOut(kv_label),
            FadeOut(output_box),
            FadeOut(output_arrow),
            run_time=2,
        )
        self.wait(1)


class FocusedPipelineView(MovingCameraScene):
    """
    Alternative version with more dramatic camera movements and zooms.
    Focuses on individual components with extreme close-ups.
    """

    def construct(self):
        # ===== SETUP COMPONENTS =====

        # Model tokens (left-top)
        model_tokens = ModelTokens(num_models=3, abbreviated=True)
        model_tokens.scale(0.7)
        model_tokens.move_to(LEFT * 4 + UP * 2)

        # Dataset tokens (left-bottom)
        dataset_tokens = DatasetTokens(num_samples=3, abbreviated=True)
        dataset_tokens.scale(0.7)
        dataset_tokens.move_to(LEFT * 4 + DOWN * 2)

        # Transformer (right center)
        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.55)
        transformer.move_to(RIGHT * 3 + UP * 0)

        # Arrows
        arrow_q = Arrow(
            model_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + UP * 0.8,
            buff=0.2,
            stroke_width=2,
            color=MATERIAL_GREEN,
        )
        q_label = MonospaceText("Q", font_size=16, color=MATERIAL_GREEN).next_to(
            arrow_q, UP, buff=0.1
        )

        arrow_kv = Arrow(
            dataset_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + DOWN * 0.8,
            buff=0.2,
            stroke_width=2,
            color=MATERIAL_RED,
        )
        kv_label = MonospaceText("K, V", font_size=16, color=MATERIAL_RED).next_to(
            arrow_kv, DOWN, buff=0.1
        )

        # ===== ANIMATION SEQUENCE =====

        # Start with extreme wide view
        self.camera.frame.set(width=25)

        # Create all components
        self.play(
            Create(model_tokens),
            Create(dataset_tokens),
            Create(transformer),
            Create(arrow_q),
            Create(q_label),
            Create(arrow_kv),
            Create(kv_label),
            run_time=2,
        )
        self.wait(1)

        # Zoom to model tokens
        self.play(
            self.camera.frame.animate.move_to(model_tokens.get_center()).set(width=6),
            run_time=1.5,
        )
        self.wait(1)

        # Pulse model tokens
        self.play(
            model_tokens.animate.scale(1.3),
            run_time=0.6,
        )
        self.play(
            model_tokens.animate.scale(1/1.3),
            run_time=0.6,
        )
        self.wait(1)

        # Zoom to dataset tokens
        self.play(
            self.camera.frame.animate.move_to(dataset_tokens.get_center()).set(width=6),
            run_time=1.5,
        )
        self.wait(1)

        # Pulse dataset tokens
        self.play(
            dataset_tokens.animate.scale(1.3),
            run_time=0.6,
        )
        self.play(
            dataset_tokens.animate.scale(1/1.3),
            run_time=0.6,
        )
        self.wait(1)

        # Zoom to transformer
        self.play(
            self.camera.frame.animate.move_to(transformer.get_center()).set(width=8),
            run_time=1.5,
        )
        self.wait(1)

        # Pulse transformer
        self.play(
            transformer.animate.scale(1.2),
            run_time=0.6,
        )
        self.play(
            transformer.animate.scale(1/1.2),
            run_time=0.6,
        )
        self.wait(1)

        # Zoom back to full view
        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=14),
            run_time=2,
        )
        self.wait(2)

        # Fade out
        self.play(
            FadeOut(model_tokens),
            FadeOut(dataset_tokens),
            FadeOut(transformer),
            FadeOut(arrow_q),
            FadeOut(q_label),
            FadeOut(arrow_kv),
            FadeOut(kv_label),
            run_time=2,
        )
        self.wait(1)


class ExpandingPipelineView(MovingCameraScene):
    """
    Dynamic scene that zooms into tokens and expands them into full pipelines.

    Flow:
    1. Show initial layout (model tokens, dataset tokens, transformer)
    2. Zoom into model token → replace with ModelPipeline → animate
    3. Zoom out
    4. Zoom into dataset token → replace with DatasetPipeline → animate
    5. Final zoom out to show complete system
    """

    def construct(self):
        # ===== PHASE 1: INITIAL SETUP =====
        # Initial camera: wide view
        self.camera.frame.set(width=20)

        # Create simple tokens (reuse FocusedPipelineView layout)
        model_tokens = ModelTokens(num_models=3, abbreviated=True)
        model_tokens.scale(0.9)
        model_tokens.move_to(LEFT * 4 + UP * 1.5)

        dataset_tokens = DatasetTokens(num_samples=3, abbreviated=True)
        dataset_tokens.scale(0.9)
        dataset_tokens.move_to(LEFT * 4 + DOWN * 1.5)
        transformer = Transformer(show_fc_layer=True)
        transformer.scale(0.55)
        transformer.move_to(RIGHT * 3)

        # Arrows (store as self for later updates)
        self.arrow_q = Arrow(
            model_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + UP * 0.8,
            buff=0.2,
            stroke_width=2,
            color=MATERIAL_GREEN,
        )
        q_label = MonospaceText("Q", font_size=16, color=MATERIAL_GREEN)
        q_label.next_to(self.arrow_q, UP, buff=0.1)

        self.arrow_kv = Arrow(
            dataset_tokens.get_right() + RIGHT * 0.2,
            transformer.get_left() + DOWN * 0.8,
            buff=0.2,
            stroke_width=2,
            color=MATERIAL_RED,
        )
        kv_label = MonospaceText("K, V", font_size=16, color=MATERIAL_RED)
        kv_label.next_to(self.arrow_kv, DOWN, buff=0.1)

        # Create all components
        self.play(
            Create(model_tokens),
            Create(dataset_tokens),
            Create(transformer),
            Create(self.arrow_q),
            Create(q_label),
            Create(self.arrow_kv),
            Create(kv_label),
            run_time=2,
        )
        self.wait(1)

        # ===== PHASE 2: MODEL TOKEN → MODELPIPELINE =====
        # Store original position for alignment
        model_token_pos = model_tokens.get_center()

        # Zoom into model token
        self.play(
            self.camera.frame.animate.move_to(model_token_pos).set(width=model_tokens.width * 2),
            run_time=2
        )
        self.wait(0.5)

        # Create ModelPipeline (initially invisible)
        model_pipeline = ModelPipeline()
        model_pipeline.scale(1)
        model_pipeline.set_opacity(0)

        # Position so final token aligns with original token position
        offset = model_tokens.get_right() - model_pipeline.model_token.get_right()
        model_pipeline.shift(offset)

        self.add(model_pipeline)

        # Fade transition: dim token → show pipeline
        self.play(
            model_tokens.animate.set_opacity(0),
            model_pipeline.animate.set_opacity(1),
            dataset_tokens.animate.shift(DOWN * 1.0),  # Adjust dataset tokens down
            self.arrow_kv.animate.put_start_and_end_on(
                self.arrow_kv.get_start() + DOWN * 1.0,
                self.arrow_kv.get_end()
            ),
            kv_label.animate.next_to(self.arrow_kv, DOWN, buff=0.1),
            self.camera.frame.animate.move_to(model_pipeline.get_center()).set(width=model_pipeline.width * 1.2),
            run_time=1.5
        )
        self.remove(model_tokens)

        # Update arrow to point from pipeline's final token
        new_arrow_start = model_pipeline.model_token.get_right() + RIGHT * 0.2
        self.play(
            self.arrow_q.animate.put_start_and_end_on(
                new_arrow_start,
                self.arrow_q.get_end()
            ),
            run_time=1
        )

        # Animate the model pipeline
        model_pipeline.animate_forward(self, run_time=12)
        self.wait(1)

        # ===== PHASE 3: ZOOM OUT =====
        # Return to bird's eye view
        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=20),
            run_time=2
        )
        self.wait(1)

        # ===== PHASE 4: DATASET TOKEN → DATASETPIPELINE =====
        # Store original position
        dataset_token_pos = dataset_tokens.get_center()

        # Zoom into dataset token
        self.play(
            self.camera.frame.animate.move_to(dataset_token_pos).set(width=dataset_tokens.width * 2),
            run_time=2
        )
        self.wait(0.5)

        # Create DatasetPipeline (initially invisible)
        dataset_pipeline = DatasetPipeline()
        dataset_pipeline.scale(0.9)
        dataset_pipeline.set_opacity(0)

        # Position so final tokens align with original position
        # Use rightmost token from dataset_tokens for alignment
        dataset_pipeline.next_to(model_pipeline, DOWN, buff=1.0, aligned_edge=RIGHT)
        self.add(dataset_pipeline)

        # Fade transition
        self.play(
            dataset_tokens.animate.set_opacity(0),
            dataset_pipeline.animate.set_opacity(1),
            self.camera.frame.animate.move_to(dataset_pipeline.get_center()).set(width=dataset_pipeline.width * 1.2),
            run_time=1.5
        )
        self.remove(dataset_tokens)

        # Update arrow to point from pipeline's final tokens
        new_arrow_start = dataset_pipeline.dataset_tokens.get_right() + RIGHT * 0.2
        self.play(
            self.arrow_kv.animate.put_start_and_end_on(
                new_arrow_start,
                self.arrow_kv.get_end()
            ),
            kv_label.animate.next_to(self.arrow_kv, DOWN, buff=0.1),
            run_time=1
        )

        # Animate the dataset pipeline
        dataset_pipeline.animate_forward(self, run_time=10)
        self.wait(1)

        # ===== PHASE 5: FINAL ZOOM OUT =====
        # Final wide view showing complete system
        width = abs(min(dataset_pipeline.get_left()[0], model_pipeline.get_left()[0]) - transformer.get_right()[0])
        midpoint_x = (min(dataset_pipeline.get_left()[0], model_pipeline.get_left()[0]) + transformer.get_right()[0]) / 2
        midpoint = np.array([midpoint_x, 0, 0])


        self.play(
            self.camera.frame.animate.move_to(midpoint).set(width=width * 1.2),
            run_time=2
        )
        self.wait(2)

        # Highlight the flow
        self.play(
            self.arrow_q.animate.set_stroke(width=4),
            run_time=0.5
        )
        self.play(
            self.arrow_q.animate.set_stroke(width=2),
            run_time=0.5
        )
        self.play(
            self.arrow_kv.animate.set_stroke(width=4),
            run_time=0.5
        )
        self.play(
            self.arrow_kv.animate.set_stroke(width=2),
            run_time=0.5
        )

        self.wait(2)

        # Fade out
        self.play(
            FadeOut(VGroup(model_pipeline, dataset_pipeline, transformer,
                          self.arrow_q, q_label, self.arrow_kv, kv_label)),
            run_time=2
        )
        self.wait(1)
