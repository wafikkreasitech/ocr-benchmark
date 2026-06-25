"""Standalone OCR benchmark for the ai4db backend.

Mirrors ai4db.ocr.pipeline.OCRPipeline but preserves bounding boxes + scores
so the benchmark can compute CER/WER and detection F1.
"""

__version__ = "0.1.0"