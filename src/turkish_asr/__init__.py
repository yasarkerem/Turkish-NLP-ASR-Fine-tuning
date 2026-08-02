"""Turkish ASR fine-tuning and evaluation toolkit."""

__version__ = "0.1.0"

from turkish_asr.normalize import TurkishNormalizer, turkish_lower

__all__ = ["TurkishNormalizer", "turkish_lower", "__version__"]
