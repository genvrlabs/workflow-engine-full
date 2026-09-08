"""
Shared project folder — asks the user once, on server startup, which
folder to use for saving all generated files (tensors, images).
Keeps everything local, so nothing needs to be uploaded anywhere.
"""

import os
import tkinter as tk
from tkinter import filedialog

_project_folder = None


def get_project_folder() -> str:
    """Returns the chosen project folder, prompting the user if not set yet."""
    global _project_folder

    if _project_folder is not None:
        return _project_folder

    root = tk.Tk()
    root.withdraw()  # hide the empty main tkinter window, we only want the dialog
    root.attributes("-topmost", True)  # make sure the dialog appears in front

    chosen = filedialog.askdirectory(title="Choose a folder where your generated images and files will be saved")
    root.destroy()

    if not chosen:
        raise RuntimeError("No project folder selected. Please restart and choose a folder.")

    _project_folder = chosen
    os.makedirs(_project_folder, exist_ok=True)
    return _project_folder