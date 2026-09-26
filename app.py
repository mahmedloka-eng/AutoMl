"""
Gradio UI for Auto_Ml.
Two tabs: Upload & Explore, Train.
"""
import logging
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from config import Config
from data_utils import (
    load_and_store_data, get_shape, detect_problem_type,
)
from model_moka import get_model_moka                     # ← renamed
from pipeline_orchestrator import run_full_pipeline, build_moka_for_target

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("automl.app")


# =============================================================================
# Session
# =============================================================================
class Session:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.state: Dict[str, Any] = {}


SESSION = Session()


# =============================================================================
# Tab 1 — Upload & Explore
# =============================================================================
def on_upload(file):
    df, msg = load_and_store_data(file)
    if df is None:
        return None, gr.update(choices=[], value=None), msg

    SESSION.df = df
    return (
        df.head(20),
        gr.update(choices=df.columns.tolist(), value=df.columns[-1]),
        get_shape(df),
    )


def on_correlation():
    df = SESSION.df
    if df is None:
        return None
    numeric = df.select_dtypes("number")
    if numeric.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def on_distributions():
    df = SESSION.df
    if df is None:
        return None
    numeric = df.select_dtypes("number").iloc[:, :9]
    if numeric.empty:
        return None
    n = numeric.shape[1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flatten(), numeric.columns):
        sns.histplot(numeric[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# =============================================================================
# Tab 2 — Train
# =============================================================================
def on_target_change(target):
    if SESSION.df is None or not target:
        return "Regression", gr.update(choices=["all"], value="all")

    y = SESSION.df[target]
    task_type, moka = build_moka_for_target(y)                
    task_label = "Regression" if task_type == "regression" else "Classification"
    return task_label, gr.update(choices=["all"] + list(moka.keys()), value="all")


def on_task_override(task_label, target):
    problem = "regression" if task_label == "Regression" else "classification"
    moka = get_model_moka(problem)                            
    return gr.update(choices=["all"] + list(moka.keys()), value="all")


def run_pipeline(target, task_label, model_choice, do_tune):
    empty_out = (None, None, None, None, None, None)

    if SESSION.df is None or not target:
        return empty_out[:-1] + ("Upload a dataset first.",)

    task_type = "regression" if task_label == "Regression" else "classification"

    try:
        result = run_full_pipeline(
            df=SESSION.df,
            target=target,
            task_type=task_type,
            model_choice=model_choice,
            do_tune=do_tune,
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        return empty_out[:-1] + (f"Pipeline failed: {e}",)

    # Stash for dropdown handlers
    SESSION.state["result_by_name"] = result["result_by_name"]
    SESSION.state["bundle_paths"] = result["bundle_paths"]
    SESSION.state["all_results"] = result["all_results"]

    model_names_list = [r["name"] for r in result["all_results"]]
    winner_name = result["winner_name"]

    return (
        result["leaderboard"],
        gr.update(choices=model_names_list, value=winner_name),
        result["winner_fig"],
        gr.update(choices=model_names_list, value=winner_name),
        result["bundle_paths"].get(winner_name),
        result["status"],
    )


def on_model_view_pick(model_name):
    if not model_name:
        return None
    r = SESSION.state.get("result_by_name", {}).get(model_name)
    return r.get("test_fig") if r else None


def on_model_download_pick(model_name):
    if not model_name:
        return None
    return SESSION.state.get("bundle_paths", {}).get(model_name)


# =============================================================================
# UI
# =============================================================================
with gr.Blocks(title="Auto_Ml") as demo:
    gr.Markdown("# 🧪 Auto_Ml")
    gr.Markdown(
        "Upload a CSV → explore → pick a target → run Auto_Ml. "
        "**Every** trained model is saved as a `.pkl` (in `models/`) and logged "
        "to MLflow (in `experiments/`). View or download any model below."
    )

    with gr.Tab("1. Upload & Explore"):
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        preview = gr.Dataframe(label="Preview (first 20 rows)")
        shape_out = gr.Textbox(label="Shape", interactive=False)
        with gr.Row():
            corr_btn = gr.Button("📊 Correlation Heatmap")
            dist_btn = gr.Button("📈 Distributions")
        with gr.Row():
            corr_plot = gr.Plot(label="Correlation")
            dist_plot = gr.Plot(label="Distributions")

    with gr.Tab("2. Train"):
        target_dd = gr.Dropdown(label="Target column", choices=[])
        task_radio = gr.Radio(
            choices=["Regression", "Classification"],
            value="Regression",
            label="Task type (auto-detected — override if wrong)",
        )
        model_dd = gr.Dropdown(label="Models to train", choices=["all"], value="all")
        tune_cb = gr.Checkbox(
            label="Tune every trained model (adds tuned rows to leaderboard)",
            value=True,
        )
        run_btn = gr.Button("🚀 Run Auto_Ml", variant="primary")

        leaderboard_out = gr.Dataframe(label="Leaderboard (all models)")

        gr.Markdown("### 🔎 Inspect any model")
        view_picker = gr.Dropdown(label="View results for…", choices=[], value=None)
        test_plot_out = gr.Plot(label="Test Set Result")

        gr.Markdown("### 💾 Download any model")
        download_picker = gr.Dropdown(label="Download bundle for…", choices=[], value=None)
        download_out = gr.File(label="Model bundle (.pkl)")

        status_out = gr.Textbox(label="Status", lines=10)

    # Wiring
    file_in.change(on_upload, file_in, [preview, target_dd, shape_out])
    corr_btn.click(on_correlation, None, corr_plot)
    dist_btn.click(on_distributions, None, dist_plot)
    target_dd.change(on_target_change, target_dd, [task_radio, model_dd])
    task_radio.change(on_task_override, [task_radio, target_dd], model_dd)

    run_btn.click(
        run_pipeline,
        [target_dd, task_radio, model_dd, tune_cb],
        [leaderboard_out, view_picker, test_plot_out,
         download_picker, download_out, status_out],
    )
    view_picker.change(on_model_view_pick, view_picker, test_plot_out)
    download_picker.change(on_model_download_pick, download_picker, download_out)


if __name__ == "__main__":
    demo.launch()