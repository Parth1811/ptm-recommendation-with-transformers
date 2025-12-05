"""Monospace text wrapper for consistent font styling across animations."""
from color_constants import get_text_color
from manim import Text as ManimText


class MonospaceText(ManimText):
    """Text with monospace font by default.

    Uses PT Mono as primary font with Andale Mono as fallback.
    Falls back to generic Monospace if neither is available.

    All parameters from Manim's Text class are supported.
    The font parameter can be overridden per-call if needed.
    """

    def __init__(self, text, font="PT Mono", font_size=48, **kwargs):
        """Initialize MonospaceText with monospace font.

        Args:
            text: The text content to display
            font: Font family (default: "PT Mono")
            font_size: Font size (default: 48)
            **kwargs: Additional arguments passed to Manim's Text class
        """
        # Try PT Mono first
        if 'color' not in kwargs:
            kwargs['color'] = get_text_color()

        try:
            super().__init__(text, font=font, font_size=font_size, **kwargs)
        except Exception:
            # Fall back to Andale Mono
            try:
                super().__init__(text, font="Andale Mono", font_size=font_size, **kwargs)
            except Exception:
                # Final fallback to generic Monospace
                super().__init__(text, font="Monospace", font_size=font_size, **kwargs)
