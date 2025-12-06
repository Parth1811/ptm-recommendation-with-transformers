"""Moving camera scene showing the complete recommendation pipeline with dynamic camera movements."""
from birdseyeview import BirdsEyeView
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


class ModelSpiderView(MovingCameraScene):
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
        self.camera.frame.set(width=24)

        title = MonospaceText("Model Spider Architecture", font_size=64, color=get_text_color())
        title.to_edge(UP)

        self.play(Write(title), run_time=2)
        self.wait(1)

        # Create birds-eye view with cross-attention layout
        birds_eye_view = BirdsEyeView(mode="self", switch_tokens=True, show_pdf=True)
        birds_eye_view.scale(1.2)
        birds_eye_view.move_to(ORIGIN)

        # Create all components
        self.play(
            Create(birds_eye_view),
            run_time=2,
        )
        self.wait(1)

        # Extract components for easier reference from current_view
        model_tokens = birds_eye_view.current_view.model_tokens
        dataset_tokens = birds_eye_view.current_view.dataset_tokens
        transformer = birds_eye_view.current_view.transformer
        arrow_qkv = birds_eye_view.current_view.dataset_arrow
        label_qkv = birds_eye_view.current_view.label_qkv

        # ===== PHASE 2: MODEL TOKEN → MODELPIPELINE =====
        # Store original position for alignment
        model_token_pos = model_tokens.get_center()

        # Zoom into model token
        self.play(
            self.camera.frame.animate.move_to(model_token_pos).set(width=model_tokens.width * 2),
            run_time=2
        )
        self.wait(0.5)

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
        dataset_pipeline = DatasetPipeline(include_unstructured=False)
        dataset_pipeline.scale(1.8)

        # Position so final tokens align with original position
        # Use rightmost token from dataset_tokens for alignment
        dataset_pipeline.next_to(model_tokens, LEFT, buff=1.0)
        dataset_pipeline.shift(UP * (model_tokens.get_center()[1] - dataset_pipeline.dataset_tokens.get_center()[1]))
        self.add(dataset_pipeline)

        # Fade transition
        self.play(
            dataset_tokens.animate.set_opacity(0),
            # GrowFromEdge(dataset_pipeline, RIGHT),
            self.camera.frame.animate.move_to(dataset_pipeline.get_center()).set(width=dataset_pipeline.width * 1.2),
            ShowIncreasingSubsets(dataset_pipeline),
            run_time=1.5
        )
        self.remove(dataset_tokens)

        # Animate the dataset pipeline
        dataset_pipeline.animate_forward(self, run_time=10)
        self.wait(1)


        # ===== PHASE 5: Transformer animations =====
        self.play(
            self.camera.frame.animate.move_to(transformer.get_center()).set(width=transformer.width * 2),
            run_time=2
        )
        self.wait(1)

        transformer.animate_show_pdf(self, duration=2)
        self.wait(1)

        # ===== PHASE 5: FINAL ZOOM OUT =====
        # Final wide view showing complete system
        width = abs(min(dataset_pipeline.get_left()[0], model_tokens.get_left()[0]) - transformer.get_right()[0])
        midpoint_x = (min(dataset_pipeline.get_left()[0], model_tokens.get_left()[0]) + transformer.get_right()[0]) / 2
        midpoint_y = (min(dataset_pipeline.get_left()[1], model_tokens.get_left()[1]) + transformer.get_right()[1]) / 2
        midpoint = np.array([midpoint_x, midpoint_y, 0])

        new_title = MonospaceText("Model Spider Architecture", font_size=92, color=get_text_color())
        new_title.move_to(midpoint + UP * 8)

        self.play(
            Transform(title, new_title),
            # title.animate.scale(1.6),
            self.camera.frame.animate.move_to(midpoint).set(width=width * 1.1),
            run_time=2
        )
        self.wait(2)

        # Fade out
        self.play(
            FadeOut(VGroup(model_tokens, dataset_pipeline, transformer, arrow_qkv, label_qkv, title)),
            run_time=2
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
        self.camera.frame.set(width=24)

        title = MonospaceText("Model Spider Architecture", font_size=64, color=get_text_color())
        title.to_edge(UP)
        self.play(Write(title), run_time=2)
        self.wait(1)

        # Create birds-eye view with cross-attention layout
        birds_eye_view = BirdsEyeView(mode="self", show_pdf=True)
        birds_eye_view.scale(1.2)
        birds_eye_view.move_to(ORIGIN)

        # Create all components
        self.play(
            Create(birds_eye_view),
            run_time=2,
        )
        self.wait(1)

        new_title = MonospaceText("Cross-Sight Architecture", font_size=64, color=get_text_color())
        new_title.to_edge(UP)
        new_title.shift(UP * 1.7)

        self.play(
            Transform(title, new_title),
            run_time=1
        )
        birds_eye_view.toggle_attention_mode(self, duration=3)

        # Extract components for easier reference from current_view
        model_tokens = birds_eye_view.current_view.model_tokens
        dataset_tokens = birds_eye_view.current_view.dataset_tokens
        transformer = birds_eye_view.current_view.transformer
        arrow_q = birds_eye_view.current_view.model_arrow
        arrow_kv = birds_eye_view.current_view.dataset_arrow
        q_label = birds_eye_view.current_view.label_q
        kv_label = birds_eye_view.current_view.label_kv

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

        # Position so final token aligns with original token position
        offset = model_tokens.get_right() - model_pipeline.model_token.get_right()
        model_pipeline.shift(offset)

        self.add(model_pipeline)

        # Fade transition: dim token → show pipeline
        self.play(
            model_tokens.animate.set_opacity(0),
            GrowFromEdge(model_pipeline, RIGHT),
            dataset_tokens.animate.shift(DOWN * 1.0),  # Adjust dataset tokens down
            arrow_kv.animate.put_start_and_end_on(
                arrow_kv.get_start() + DOWN * 1.0,
                arrow_kv.get_end()
            ),
            kv_label.animate.next_to(arrow_kv, DOWN, buff=0.1),
            self.camera.frame.animate.move_to(model_pipeline.get_center()).set(width=model_pipeline.width * 1.2),
            run_time=1.5
        )
        self.remove(model_tokens)

        # Update arrow to point from pipeline's final token
        new_arrow_start = model_pipeline.model_token.get_right() + RIGHT * 0.2
        self.play(
            arrow_q.animate.put_start_and_end_on(
                new_arrow_start,
                arrow_q.get_end()
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

        # Position so final tokens align with original position
        # Use rightmost token from dataset_tokens for alignment
        dataset_pipeline.next_to(model_pipeline, DOWN, buff=1.0, aligned_edge=RIGHT)
        self.add(dataset_pipeline)

        # Fade transition
        self.play(
            dataset_tokens.animate.set_opacity(0),
            GrowFromEdge(dataset_pipeline, RIGHT),
            self.camera.frame.animate.move_to(dataset_pipeline.get_center()).set(width=dataset_pipeline.width * 1.2),
            run_time=1.5
        )
        self.remove(dataset_tokens)

        # Update arrow to point from pipeline's final tokens
        new_arrow_start = dataset_pipeline.dataset_tokens.get_right() + RIGHT * 0.2
        self.play(
            arrow_kv.animate.put_start_and_end_on(
                new_arrow_start,
                arrow_kv.get_end()
            ),
            kv_label.animate.next_to(arrow_kv, DOWN, buff=0.1),
            run_time=1
        )

        # Animate the dataset pipeline
        dataset_pipeline.animate_forward(self, run_time=10)
        self.wait(1)

        # ===== PHASE 5: FINAL ZOOM OUT =====
        # Final wide view showing complete system
        width = abs(min(dataset_pipeline.get_left()[0], model_pipeline.get_left()[0]) - transformer.get_right()[0])
        midpoint_x = (min(dataset_pipeline.get_left()[0], model_pipeline.get_left()[0]) + transformer.get_right()[0]) / 2
        midpoint_y = (min(dataset_pipeline.get_left()[1], model_pipeline.get_left()[1]) + transformer.get_right()[1]) / 2
        midpoint = np.array([midpoint_x, midpoint_y, 0])

        new_title = MonospaceText("Cross-Sight Architecture", font_size=64, color=get_text_color())
        new_title.move_to(midpoint + UP * 7.5)

        self.play(
            Transform(title, new_title),
            self.camera.frame.animate.move_to(midpoint + UP * 1.5).set(width=width * 1.15),
            run_time=2
        )
        self.wait(2)

        # Highlight the flow
        self.play(
            GrowArrow(arrow_q),
            GrowArrow(arrow_kv),
            run_time=3
        )
        self.wait(1)

        # self.play(
        #     GrowArrow(transformer.attention_to_fc_arrow),
        #     run_time=3
        # )

        # self.wait(2)

        # Fade out
        self.play(
            FadeOut(VGroup(model_pipeline, dataset_pipeline, transformer,
                          arrow_q, q_label, arrow_kv, kv_label, title)),
            run_time=2
        )
        self.wait(1)
