"""
Runs once when the server starts — asks the user for both the project
folder (outputs) and the models folder (local model files), together,
in one clean startup flow.
"""

from nodes.diffusion._project_folder import get_project_folder
from nodes.diffusion._models_folder import get_models_folder


def run_startup_prompts():
    """Call this once when the server starts, to ask both folder questions upfront."""
    print("=" * 60)
    print("Workflow Engine — first-time setup")
    print("=" * 60)

    project_folder = get_project_folder()
    print(f"✅ Project folder (outputs): {project_folder}")

    models_folder = get_models_folder()
    print(f"✅ Models folder: {models_folder}")

    print("=" * 60)