"""
Lets the user pick ANY image file, from anywhere, each time it's called.
Unlike the models folder, this does NOT remember a choice — every call
opens a fresh file picker.
"""

import tkinter as tk
from tkinter import filedialog


def pick_image_file() -> str:
    """Opens a file picker for the user to select an image, fresh every time."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    chosen = filedialog.askopenfilename(
        title="Choose an image to use",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")],
    )
    root.destroy()

    if not chosen:
        raise RuntimeError("No image selected.")

    return chosen