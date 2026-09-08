"""
Shared models folder — asks the user ONCE, ever, where to store all
downloaded model files. Remembers the choice permanently in a small
config file, so future server restarts don't ask again.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog

_ALLOW_AUTO_DOWNLOAD = os.environ.get("ALLOW_AUTO_DOWNLOAD", "false").lower() == "true"

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "_models_folder_config.json")

_models_folder = None


def get_models_folder() -> str:
    """Returns the permanent models folder, asking the user only the very first time."""
    global _models_folder

    if _models_folder is not None:
        return _models_folder

    # Check if we've already saved a choice from a previous run
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE, "r") as f:
            saved = json.load(f)
        saved_path = saved.get("models_folder")
        if saved_path and os.path.isdir(saved_path):
            _models_folder = saved_path
            return _models_folder

    # No valid saved choice — ask the user, this one time
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    chosen = filedialog.askdirectory(title="Choose your MODELS folder (used every time, remembered permanently)")
    root.destroy()

    if not chosen:
        raise RuntimeError("No models folder selected. Please restart and choose a folder.")

    os.makedirs(chosen, exist_ok=True)

    # Save this choice permanently for all future runs
    with open(_CONFIG_FILE, "w") as f:
        json.dump({"models_folder": chosen}, f)

    _models_folder = chosen
    return _models_folder


def get_model_path(repo_id: str) -> str:
    """Returns the local subfolder path for a specific model, e.g.
    'stabilityai/stable-diffusion-xl-base-1.0' -> '{models_folder}/stabilityai_stable-diffusion-xl-base-1.0'
    """
    folder = get_models_folder()
    safe_name = repo_id.replace("/", "_")
    return os.path.join(folder, safe_name)

def require_model_path(repo_id: str) -> str:
    """
    Returns the local path for a model, or raises a clear error
    telling the user exactly what to download and where to put it.

    If ALLOW_AUTO_DOWNLOAD=true is set (testing only), falls back to
    letting from_pretrained download directly from HuggingFace instead.
    """
    path = get_model_path(repo_id)

    if not os.path.isdir(path):
        if _ALLOW_AUTO_DOWNLOAD:
            print(f"⚠️  TESTING MODE: '{repo_id}' not found locally, falling back to HuggingFace download")
            return repo_id

        raise RuntimeError(
            f"Model not found locally.\n"
            f"Please download '{repo_id}' and place its files in:\n"
            f"  {path}\n"
            f"(with subfolders: unet/, vae/, text_encoder/, text_encoder_2/, "
            f"tokenizer/, tokenizer_2/, scheduler/ — matching HuggingFace's own layout)"
        )

    return path