"""Gradio web app for the Lab 8 mosaic generator.

Tiles are loaded ONCE at startup into a shared TileManager; each request only
builds a lightweight MosaicBuilder. Run locally with ``python app.py``. On Render, the same command serves on $PORT.
"""
from __future__ import annotations

import logging
import os
import time

import gradio as gr

from mosaic_generator import MosaicBuilder, MosaicError, TileManager, config
from mosaic_generator.metrics import compute_metrics
from mosaic_generator.utils import format_seconds

logging.basicConfig(level=logging.INFO)

# Loaded once per process, not per request.
TILE_MANAGER = TileManager(config.DEFAULT_TILE_DIR)

MATCH_LABELS = {
    "Lab 1 palette colours": "palette",
    "Measured tile colours (closer match)": "tile_mean",
}


def generate(image, grid_size, tile_count, match_label, with_ssim):
    """Build a mosaic for the web UI and report timing and similarity metrics.

    Args:
        image (np.ndarray | None): Uploaded RGB image from Gradio.
        grid_size (float): Cells across and down (slider value).
        tile_count (float): Number of tiles to use (Lab 1 "Tile Set").
        match_label (str): A key of :data:`MATCH_LABELS`.
        with_ssim (bool): Also compute SSIM (slower on large images).

    Returns:
        tuple: ``(segmented, mosaic, mse, ssim or None, performance markdown)``.

    Raises:
        gr.Error: With a readable message if the input is invalid.
    """
    if image is None:
        raise gr.Error("Please upload an image (or click one of the examples).")
    try:
        builder = MosaicBuilder(
            TILE_MANAGER,
            grid_size=int(grid_size),
            tile_count=int(tile_count),
            match_on=MATCH_LABELS.get(match_label, config.DEFAULT_MATCH_MODE),
        )
        result = builder.build(image)
        start = time.perf_counter()
        names = ("mse", "ssim") if with_ssim else ("mse",)
        scores = compute_metrics(result.processed, result.mosaic, names)
        metrics_time = time.perf_counter() - start
    except MosaicError as exc:
        raise gr.Error(str(exc)) from exc

    rows, cols = result.grid_size
    cell_h, cell_w = result.cell_size
    h, w = result.mosaic.shape[:2]
    stage_lines = "\n".join(
        f"| {name} | {format_seconds(sec)} |" for name, sec in result.timings.items()
    )
    cache = TILE_MANAGER.cache_info()
    perf = (
        f"### ⏱️ Mosaic generated in **{format_seconds(result.total_time)}**\n"
        f"Metrics: {format_seconds(metrics_time)} · {w}×{h} px · {cols}×{rows} grid "
        f"({rows * cols:,} cells of {cell_w}×{cell_h} px) · matcher: "
        f"`{builder._get_matcher().name}`\n\n"
        f"| Stage | Time |\n|---|---|\n{stage_lines}\n\n"
        f"Tiles loaded once at startup in {format_seconds(TILE_MANAGER.load_time)} · "
        f"resized-tile cache: {cache['hits']} hits / {cache['misses']} misses. "
        f"(The Lab 1 baseline took about 4 s per mosaic.)"
    )
    return result.segmented, result.mosaic, scores["mse"], scores.get("ssim"), perf


def build_demo() -> gr.Blocks:
    """Create the Gradio interface.

    Returns:
        gr.Blocks: The app, ready for ``.launch()``.
    """
    examples = sorted(p for p in config.EXAMPLES_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    with gr.Blocks(title="Image Mosaic Generator (Lab 8)") as demo:
        gr.Markdown(
            "# 🧩 Image Mosaic Generator, optimized\n"
            "Upload an image, choose a grid, and the image is rebuilt from colour tiles. "
            "Vectorized NumPy pipeline with cached tiles: typically a few milliseconds per mosaic."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="numpy", image_mode="RGB", label="Upload Image")
                grid = gr.Slider(8, 128, value=config.DEFAULT_GRID_SIZE, step=8, label="Grid Size (cells across and down)")
                tiles = gr.Slider(3, TILE_MANAGER.count, value=min(config.DEFAULT_TILE_COUNT, TILE_MANAGER.count),
                                  step=1, label="Tile Set (number of tiles)")
                match = gr.Radio(list(MATCH_LABELS), value="Lab 1 palette colours", label="Match cells against")
                ssim_box = gr.Checkbox(value=True, label="Compute SSIM (slower for large images)")
                run = gr.Button("Generate Mosaic", variant="primary")
            with gr.Column():
                perf = gr.Markdown("Processing time will appear here.")
                with gr.Row():
                    mse_out = gr.Number(label="MSE", precision=2)
                    ssim_out = gr.Number(label="SSIM", precision=4)
        with gr.Row():
            seg_out = gr.Image(label="Segmented Image (cell averages)")
            mosaic_out = gr.Image(label="Mosaic Image")

        inputs = [image_in, grid, tiles, match, ssim_box]
        outputs = [seg_out, mosaic_out, mse_out, ssim_out, perf]
        if examples:
            gr.Examples(examples=[[str(p), 32, 9, "Lab 1 palette colours", True] for p in examples],
                        inputs=inputs, label="Example images")
        run.click(generate, inputs=inputs, outputs=outputs)
    return demo


demo = build_demo()

if __name__ == "__main__":
    # Hosts such as Render assign the port through $PORT and need 0.0.0.0;
    # locally this falls back to Gradio's usual http://127.0.0.1:7860.
    port = os.environ.get("PORT")
    if port:
        demo.launch(server_name="0.0.0.0", server_port=int(port))
    else:
        demo.launch()
