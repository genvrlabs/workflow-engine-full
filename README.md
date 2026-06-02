# GenVR Workflow Engine

A Python/FastAPI workflow engine that exposes pluggable **nodes** over HTTP. Built to integrate with the [GenVR Workflow Designer](https://genvr.com).

---

## Quick start

### 1. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

> Your prompt will show `(.venv)` when the environment is active. Run `deactivate` to exit it.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### 4. ComfyUI custom-node installer (port 8001)

Separate service to install [ComfyUI custom nodes](https://docs.comfy.org/custom-nodes/backend/lifecycle) from GitHub or a local path. Packages land in `comfyui/installed/` and are linked into `comfyui/custom_nodes/` for ComfyUI to load (`NODE_CLASS_MAPPINGS` in each package `__init__.py`).

```bash
python run_comfyui_installer.py
```

Web UI: `http://localhost:8001/` — enter a GitHub URL or local path and click **Install**. The installer runs `pip install` for dependencies from `requirements.txt`, `pyproject.toml`, and Python imports in the node code (e.g. PoseNode → Pillow, numpy, torch). A snapshot is saved as `requirements-genvr.txt` in the install folder.

API docs: `http://localhost:8001/docs`

**Install from GitHub (example [PoseNode](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet/blob/master/PoseNode/pose_node.py)):**

```bash
curl -X POST http://localhost:8001/comfyui/install ^
  -H "Content-Type: application/json" ^
  -d "{\"source\": \"https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet/blob/master/PoseNode/pose_node.py\"}"
```

**List installations:**

```bash
curl http://localhost:8001/comfyui/installed
```

Point your ComfyUI install’s `custom_nodes` folder at this repo’s `comfyui/custom_nodes` (or copy/symlink nodes from there), then restart ComfyUI.

Installed ComfyUI nodes also appear in the **workflow engine** node list (`GET http://localhost:8000/nodes`) under category `comfyui`, e.g. `comfyui.<package>.PoseNode`. Restart the workflow engine (`uvicorn main:app`) after installing new ComfyUI nodes. Execution inside the workflow engine requires ComfyUI’s Python stack (`torch`, `folder_paths`, etc.); otherwise run the node in ComfyUI itself.

---

## API

### List nodes
```
GET /nodes
```
Returns all registered nodes with their `metadata`, `inputs`, and `outputs`.

### Execute a single node
```
POST /nodes/{node_type}/execute
```
```json
{
  "uid": "user-id",
  "token": "auth-token",
  "inputs": { "a": 3, "b": 4, "c": 5 }
}
```
Example — add three numbers:
```
POST /nodes/arithmetic.add_three/execute
```

### Execute a full workflow graph
```
POST /workflow/execute
```
```json
{
  "uid": "user-id",
  "token": "auth-token",
  "nodes": [
    { "id": "n1", "type": "arithmetic.add_two", "data": { "a": 10, "b": 5 } },
    { "id": "n2", "type": "arithmetic.multiply",  "data": { "b": 2 } }
  ],
  "edges": [
    {
      "source": "n1", "sourceHandle": "result",
      "target": "n2", "targetHandle": "a"
    }
  ]
}
```
The engine resolves nodes in **topological order**. Each node's inputs are merged from:
1. Its own `data` (static / default values)
2. The wired outputs of upstream nodes (edges)

---

## Node contract

Every node is a Python module (file) placed anywhere under `nodes/`. It must export:

| Export | Type | Description |
|--------|------|-------------|
| `metadata` | `dict` | `display_name`, `description`, `category`, `color` |
| `inputs` | `list[dict]` | Each entry: `{"var_name", "display_name", "type"}` |
| `outputs` | `list[dict]` | Each entry: `{"var_name", "display_name", "type"}` |
| `execute` | `async fn` | `(uid, token, inputs) → dict` |

Example — adding three numbers (`nodes/arithmetic/add_three.py`):

```python
metadata = {
    "display_name": "Add Three Numbers",
    "description": "Adds three numbers and returns the result.",
    "category": "arithmetic",
    "color": "green",
}

inputs = [
    {"var_name": "a", "display_name": "Number A", "type": "number"},
    {"var_name": "b", "display_name": "Number B", "type": "number"},
    {"var_name": "c", "display_name": "Number C", "type": "number"},
]

outputs = [
    {"var_name": "result", "display_name": "Result", "type": "number"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    a = float(inputs.get("a", 0))
    b = float(inputs.get("b", 0))
    c = float(inputs.get("c", 0))
    return {"result": a + b + c}
```

The file path maps to the `node_type`:  
`nodes/arithmetic/add_three.py` → `arithmetic.add_three`

---

## Adding new nodes

1. Create a `.py` file anywhere under `nodes/` (subdirectories are fine).
2. Export `metadata`, `inputs`, `outputs`, and `async def execute(...)`.
3. Restart the server — the registry auto-discovers all modules.

No registration step needed.

---

## 🤖 LLM instructions — creating a custom node

> Copy this section verbatim into your prompt when asking an LLM to add a new node.

---

### SYSTEM PROMPT — GenVR Workflow Engine custom node

You are adding a new node to the **GenVR Workflow Engine** — a Python/FastAPI server that exposes executable nodes over HTTP to the GenVR visual workflow designer.

#### Rules you must follow

1. **One file = one node.** Create a single `.py` file inside `nodes/<category>/`.  
   - The category folder must contain an empty `__init__.py` (create it if absent).  
   - The file path determines the node type: `nodes/text/uppercase.py` → node type `text.uppercase`.  
   - Use `snake_case` for both folder and file names.

2. **Every node file must export exactly these four names at module level:**

```python
metadata: dict        # who this node is
inputs:   list[dict]  # what it accepts
outputs:  list[dict]  # what it returns
execute               # async function that does the work
```

3. **`metadata` shape** — all keys are required:

```python
metadata = {
    "display_name": str,   # shown in the designer palette, title-case, e.g. "Uppercase Text"
    "description":  str,   # one sentence, shown as a tooltip
    "category":     str,   # matches the folder name, e.g. "text"
    "color":        str,   # one of: "green" | "blue" | "red" | "orange" | "purple" | "gray"
}
```

4. **`inputs` and `outputs` shape** — each element is a dict with three required keys:

```python
{"var_name": str, "display_name": str, "type": str}
```

- `var_name` — `snake_case`, used as the dict key in `execute()`  
- `display_name` — human-readable label shown in the designer  
- `type` — one of: `"number"` `"text"` `"boolean"` `"any"` `"list"` `"dict"`  
- `batch` — optional. `true` = run once per list item (URLs or `{name, uri, type}` assets); `false` = treat the whole list as one input (e.g. stitch `parts`, concat `input_urls`). Media ports (`*_url`, `input_url`) auto-batch when given a list unless `batch: false`.

The engine deconstructs inputs before `execute()` (see `nodes/input_utils.py`): asset objects become URL strings; batched runs return list outputs plus `batch_count`.

5. **`execute` signature — never change it:**

```python
async def execute(uid: str, token: str, inputs: dict) -> dict:
```

- `uid` and `token` are auth credentials forwarded from the designer — pass them to any downstream API calls you make; do not validate them yourself.  
- `inputs` is a `dict` whose keys are the `var_name` strings from `inputs` list above.  
- Always call `inputs.get("var_name", <sensible_default>)` — never assume a key is present.  
- Return a `dict` whose keys exactly match the `var_name` strings in the `outputs` list.  
- Raise a plain `ValueError` for bad input (the engine turns it into a 422 response).  
- Raise any other exception for unexpected errors (the engine turns it into a 500 response).  
- **Never** return `None` — return an empty dict `{}` if there is genuinely nothing to return.

6. **No side-effects on import.** All logic must be inside `execute()`. The file is imported at server startup; heavy imports (torch, cv2, etc.) should be done at the top of the file but must not execute anything.

7. **No registration step.** The registry auto-discovers every `.py` file under `nodes/` that exports all four required names. Just create the file and restart the server.

---

#### Minimal template — copy and fill in

```python
# nodes/<category>/<node_name>.py

metadata = {
    "display_name": "<Human Readable Name>",
    "description":  "<One sentence describing what this node does.>",
    "category":     "<category>",
    "color":        "green",   # green | blue | red | orange | purple | gray
}

inputs = [
    {"var_name": "input_a", "display_name": "Input A", "type": "text"},
    # add more as needed
]

outputs = [
    {"var_name": "result", "display_name": "Result", "type": "text"},
    # add more as needed
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_a = inputs.get("input_a", "")

    # ── your logic here ──────────────────────────────────────────
    result = input_a  # replace with real logic
    # ─────────────────────────────────────────────────────────────

    return {"result": result}
```

---

#### Worked examples

**Example 1 — pure computation (no external calls)**

```python
# nodes/text/uppercase.py

metadata = {
    "display_name": "Uppercase Text",
    "description":  "Converts a text string to uppercase.",
    "category":     "text",
    "color":        "blue",
}

inputs  = [{"var_name": "text",   "display_name": "Text",   "type": "text"}]
outputs = [{"var_name": "result", "display_name": "Result", "type": "text"}]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    text = str(inputs.get("text", ""))
    return {"result": text.upper()}
```

**Example 2 — external HTTP call, uses uid/token**

```python
# nodes/ai/summarise.py
import httpx

metadata = {
    "display_name": "Summarise Text",
    "description":  "Calls an external API to summarise a passage of text.",
    "category":     "ai",
    "color":        "purple",
}

inputs = [
    {"var_name": "text",       "display_name": "Text",        "type": "text"},
    {"var_name": "max_words",  "display_name": "Max Words",   "type": "number"},
]
outputs = [
    {"var_name": "summary", "display_name": "Summary", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    text      = str(inputs.get("text", ""))
    max_words = int(inputs.get("max_words", 50))

    if not text.strip():
        raise ValueError("'text' must not be empty")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/summarise",
            json={"text": text, "max_words": max_words},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

    return {"summary": data["summary"]}
```

**Example 3 — multiple outputs**

```python
# nodes/text/split_words.py

metadata = {
    "display_name": "Split Words",
    "description":  "Splits a sentence into a word list and returns the word count.",
    "category":     "text",
    "color":        "green",
}

inputs = [
    {"var_name": "sentence",  "display_name": "Sentence",  "type": "text"},
]
outputs = [
    {"var_name": "words",  "display_name": "Words",      "type": "list"},
    {"var_name": "count",  "display_name": "Word Count", "type": "number"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    sentence = str(inputs.get("sentence", ""))
    words = sentence.split()
    return {"words": words, "count": len(words)}
```

---

#### Checklist before finishing

- [ ] File is at `nodes/<category>/<node_name>.py`
- [ ] `nodes/<category>/__init__.py` exists (even if empty)
- [ ] All four module-level names exported: `metadata`, `inputs`, `outputs`, `execute`
- [ ] `metadata` has `display_name`, `description`, `category`, `color`
- [ ] Every port dict has `var_name`, `display_name`, `type`
- [ ] `execute` is `async def execute(uid, token, inputs) -> dict`
- [ ] All `inputs.get()` calls have a sensible default
- [ ] Return dict keys match output `var_name` values exactly
- [ ] No logic runs at import time
- [ ] Server restart will auto-register the node (no other files to edit)

---

## Calling from GenVR Workflow Designer

In the workflow designer, add a node of a custom type (e.g. `python_engine`) that:

1. On load, calls `GET /nodes` to populate the node palette.
2. On execution, calls `POST /nodes/{node_type}/execute` with the resolved inputs.

Alternatively, send the entire graph to `POST /workflow/execute` and let the engine handle ordering and data-passing.

---

## GenVR file upload (nodes that work with media)

Any node that produces a file (video, audio, image) must upload it to GenVR blob storage and return the URL — **never** return a local file path. The container is **private**, so the returned URL must include the SAS token.

### How it works (`nodes/ffmpeg/_utils.py`)

The helper `upload_file(uid, token, local_path)` does this in two steps:

**Step 1 — get a SAS upload URL**
```
POST https://api.genvrresearch.com/api/upload/sas-url
Authorization: Bearer <token>
Content-Type: application/json

{ "userId": "<uid>", "fileName": "output.mp4", "category": "imagegen" }
```
Response fields you need:
| Field | Used for |
|---|---|
| `uploadUrl` | PUT target + **contains the SAS token as the query string** |
| `account` | Azure storage account name |
| `container` | Blob container name |
| `blobPath` | Path inside the container |

**Step 2 — PUT the file to Azure**
```
PUT <uploadUrl>
x-ms-blob-type: BlockBlob
Content-Type: <mime type>
```
Expected response: `HTTP 201 Created` (empty body).

**Step 3 — build the output URL (with SAS token)**

The container is private, so callers cannot access a bare blob URL. Append the SAS token from `uploadUrl`'s query string to the blob URL:

```python
sas_token = urllib.parse.urlparse(upload_url).query
blob_url = f"https://{account}.blob.core.windows.net/{container}/{blob_path}?{sas_token}"
```

> **Do not** return `https://{account}.blob.core.windows.net/{container}/{blob_path}` without the `?{sas_token}` — the container is private and the URL will return 403.

### Rules for nodes that handle files

1. **Inputs are URLs, not paths.** Download them with `download_to_tempfile(url, suffix)`.
2. **Outputs are URLs, not paths.** Upload with `upload_file(uid, token, local_path)` and return the result.
3. **Always clean up temp files** in a `finally` block, even on error.
4. **Pass `uid` and `token` through** — they come in from `execute(uid, token, inputs)` and are needed for the SAS URL request.

### Minimal pattern for a file-producing node

```python
import os, tempfile, subprocess
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    if not input_url:
        raise ValueError("input_url is required")

    ext = url_suffix(input_url) or ".mp4"
    tmp_in  = download_to_tempfile(input_url, suffix=ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out.close()

    try:
        # run your processing (ffmpeg, PIL, etc.)
        subprocess.run(["ffmpeg", "-y", "-i", tmp_in, tmp_out.name], check=True)

        # upload and get back a URL with SAS token
        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
```

### SAS token expiry

SAS URLs are valid for ~1 hour. If you store an output URL and try to use it later, it may return 403. Always re-request a fresh upload if auth errors occur.

---

## OpenCV nodes

100 VFX-grade image-processing nodes built on OpenCV. All nodes follow the same pattern:

- **Input**: one or more image URLs (`input_url`, `background_url`, etc.)
- **Output**: one or more image URLs (`output_url`, `mask_url`, etc.)
- Images are downloaded from the URL, processed in memory, uploaded to GenVR blob storage, and the SAS-signed URL is returned.

### Dependency

```
opencv-contrib-python>=4.9.0
numpy>=1.26.0
```

Already listed in `requirements.txt`. The `oil_painting` node requires the `-contrib` variant for `cv2.xphoto.oilPainting`.

### Shared utility — `nodes/opencv/_utils.py`

All OpenCV nodes import from this shared helper:

```python
from nodes.opencv._utils import (
    load_image_from_url,   # download URL → np.ndarray
    save_and_upload,       # np.ndarray → upload → return SAS URL
    parse_color,           # "B,G,R" string → (int, int, int) tuple
    odd,                   # round up to nearest odd int (kernel sizes)
    ensure_bgr,            # BGRA/GRAY → BGR
    ensure_bgra,           # BGR/GRAY → BGRA (adds alpha channel)
)
```

The upload path reuses `nodes/ffmpeg/_utils.py` — same SAS token logic, same `upload_file()` helper.

### Node catalogue

Node type = `opencv.<category>.<node_name>` (e.g. `opencv.color.clahe`).

#### color (15 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.color.adjust_brightness_contrast` | Brightness / Contrast | `brightness` (-100→100), `contrast` (-100→100) |
| `opencv.color.adjust_hue_saturation` | Hue / Saturation | `hue` (-180→180), `saturation` (-100→100), `lightness` (-100→100) |
| `opencv.color.color_balance` | Color Balance | `cyan_red`, `magenta_green`, `yellow_blue` (-100→100 each) |
| `opencv.color.gamma_correct` | Gamma Correct | `gamma` (0.1→5.0) |
| `opencv.color.auto_levels` | Auto Levels | _(no extra params)_ |
| `opencv.color.histogram_equalize` | Histogram Equalize | _(no extra params)_ |
| `opencv.color.clahe` | CLAHE | `clip_limit`, `tile_size` |
| `opencv.color.temperature_tint` | Temperature / Tint | `temperature` (-100→100), `tint` (-100→100) |
| `opencv.color.vibrance` | Vibrance | `vibrance` (-100→100) |
| `opencv.color.posterize` | Posterize | `levels` (2→8) |
| `opencv.color.solarize` | Solarize | `threshold` (0→255) |
| `opencv.color.invert_colors` | Invert Colors | _(no extra params)_ |
| `opencv.color.desaturate` | Desaturate | `amount` (0→100) |
| `opencv.color.curves` | Curves | `shadows`, `midtones`, `highlights` (0→255 output points) |
| `opencv.color.split_rgb` | Split RGB | _(no extra params)_ → outputs `red_url`, `green_url`, `blue_url` |

#### blur (10 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.blur.gaussian_blur` | Gaussian Blur | `kernel_size`, `sigma` |
| `opencv.blur.box_blur` | Box Blur | `kernel_size` |
| `opencv.blur.median_blur` | Median Blur | `kernel_size` |
| `opencv.blur.bilateral_filter` | Bilateral Filter | `diameter`, `sigma_color`, `sigma_space` |
| `opencv.blur.motion_blur` | Motion Blur | `kernel_size`, `angle` |
| `opencv.blur.radial_blur` | Radial Blur | `strength`, `center_x`, `center_y` |
| `opencv.blur.sharpen` | Sharpen | `strength` |
| `opencv.blur.unsharp_mask` | Unsharp Mask | `radius`, `amount`, `threshold` |
| `opencv.blur.tilt_shift` | Tilt Shift | `focus_y`, `focus_width`, `blur_strength` |
| `opencv.blur.glow_blur` | Glow Blur | `blur_radius`, `intensity` |

#### key (8 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.key.chroma_key` | Chroma Key | `key_color` (green/blue/custom), `hue_range`, `saturation_min`, `spill_suppress` → `mask_url` + `output_url` |
| `opencv.key.luma_key` | Luma Key | `luma_min`, `luma_max` → `mask_url` + `output_url` |
| `opencv.key.difference_key` | Difference Key | `background_url`, `threshold`, `softness` → `mask_url` + `output_url` |
| `opencv.key.range_key` | Range Key | `low_color` (B,G,R), `high_color` (B,G,R) → `mask_url` + `output_url` |
| `opencv.key.threshold_key` | Threshold Key | `threshold`, `mode` (binary/otsu) → `mask_url` |
| `opencv.key.adaptive_threshold` | Adaptive Threshold | `block_size`, `C`, `method` → `mask_url` |
| `opencv.key.feather_matte` | Feather Matte | `mask_url`, `feather_radius` |
| `opencv.key.erode_dilate_matte` | Erode / Dilate Matte | `mask_url`, `erode`, `dilate`, `kernel_size` |

#### composite (10 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.composite.composite_over` | Composite Over | `foreground_url` (BGRA), `background_url` |
| `opencv.composite.blend_add` | Blend Add | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_multiply` | Blend Multiply | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_screen` | Blend Screen | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_overlay` | Blend Overlay | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_soft_light` | Blend Soft Light | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_difference` | Blend Difference | `layer_a_url`, `layer_b_url`, `opacity` |
| `opencv.composite.blend_mix` | Blend Mix | `layer_a_url`, `layer_b_url`, `mix` (0→1) |
| `opencv.composite.apply_mask` | Apply Mask | `input_url`, `mask_url`, `invert_mask` |
| `opencv.composite.combine_masks` | Combine Masks | `mask_a_url`, `mask_b_url`, `operation` (add/subtract/multiply/intersect) |

#### transform (10 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.transform.resize_image` | Resize Image | `width`, `height`, `interpolation` |
| `opencv.transform.crop_image` | Crop Image | `x`, `y`, `width`, `height` |
| `opencv.transform.rotate_image` | Rotate Image | `angle`, `scale`, `expand` |
| `opencv.transform.flip_image` | Flip Image | `direction` (horizontal/vertical/both) |
| `opencv.transform.perspective_warp` | Perspective Warp | `src_points` (JSON), `dst_points` (JSON), `output_width`, `output_height` |
| `opencv.transform.pad_image` | Pad Image | `top`, `bottom`, `left`, `right`, `color` (B,G,R) |
| `opencv.transform.fit_image` | Fit Image | `width`, `height`, `bg_color` (B,G,R) |
| `opencv.transform.center_crop` | Center Crop | `width`, `height` |
| `opencv.transform.affine_transform` | Affine Transform | `matrix` (JSON 2×3 array) |
| `opencv.transform.tile_image` | Tile Image | `cols`, `rows` |

#### noise (5 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.noise.add_film_grain` | Add Film Grain | `intensity`, `size` |
| `opencv.noise.add_gaussian_noise` | Add Gaussian Noise | `mean`, `stddev` |
| `opencv.noise.add_salt_pepper` | Add Salt & Pepper | `amount`, `salt_ratio` |
| `opencv.noise.denoise_nlm` | Denoise NLM | `h`, `template_window`, `search_window` |
| `opencv.noise.denoise_bilateral` | Denoise Bilateral | `diameter`, `sigma_color`, `sigma_space` |

#### morph (6 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.morph.erode` | Erode | `kernel_size`, `iterations`, `shape` (rect/ellipse/cross) |
| `opencv.morph.dilate` | Dilate | `kernel_size`, `iterations`, `shape` |
| `opencv.morph.morph_open` | Morph Open | `kernel_size`, `shape` |
| `opencv.morph.morph_close` | Morph Close | `kernel_size`, `shape` |
| `opencv.morph.morph_gradient` | Morph Gradient | `kernel_size`, `shape` |
| `opencv.morph.morph_tophat` | Morph Top Hat | `kernel_size`, `shape`, `blackhat` |

#### edge (7 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.edge.canny_edge` | Canny Edge | `threshold1`, `threshold2`, `aperture` |
| `opencv.edge.sobel_edge` | Sobel Edge | `ksize`, `dx`, `dy` |
| `opencv.edge.laplacian_edge` | Laplacian Edge | `ksize` |
| `opencv.edge.find_contours` | Find Contours | `threshold`, `draw_color` (B,G,R), `thickness` |
| `opencv.edge.hough_lines` | Hough Lines | `threshold`, `min_line_length`, `max_line_gap`, `draw_color` |
| `opencv.edge.hough_circles` | Hough Circles | `min_radius`, `max_radius`, `param1`, `param2`, `draw_color` |
| `opencv.edge.harris_corners` | Harris Corners | `block_size`, `k_size`, `k`, `threshold`, `draw_color` |

#### lens (8 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.lens.vignette` | Vignette | `strength`, `radius` |
| `opencv.lens.chromatic_aberration` | Chromatic Aberration | `shift_x`, `shift_y` |
| `opencv.lens.lens_distortion` | Lens Distortion | `k1`, `k2` (barrel/pincushion coefficients) |
| `opencv.lens.depth_of_field` | Depth of Field | `focus_x`, `focus_y`, `focus_radius`, `blur_strength` |
| `opencv.lens.glow` | Glow | `blur_radius`, `intensity`, `threshold` |
| `opencv.lens.anamorphic_flare` | Anamorphic Flare | `strength`, `angle`, `color` (B,G,R) |
| `opencv.lens.film_halation` | Film Halation | `radius`, `intensity`, `color` (B,G,R) |
| `opencv.lens.scanlines` | Scanlines | `line_spacing`, `opacity`, `color` (B,G,R) |

#### style (9 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.style.cartoon` | Cartoon | `num_bilateral`, `d`, `sigma`, `edge_threshold` |
| `opencv.style.pencil_sketch` | Pencil Sketch | `sigma_s`, `sigma_r`, `shade_factor` |
| `opencv.style.emboss` | Emboss | `strength` |
| `opencv.style.pixelate` | Pixelate | `pixel_size` |
| `opencv.style.glitch_rgb` | Glitch RGB | `shift_amount`, `num_slices` |
| `opencv.style.duotone` | Duotone | `shadow_color` (B,G,R), `highlight_color` (B,G,R) |
| `opencv.style.oil_painting` | Oil Painting | `size`, `dynRatio` — requires `opencv-contrib-python` |
| `opencv.style.hdr_tonemap` | HDR Tonemap | `gamma`, `saturation` |
| `opencv.style.pixel_sort` | Pixel Sort | `threshold`, `direction` (horizontal/vertical) |

#### channel (5 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.channel.merge_channels` | Merge Channels | `blue_url`, `green_url`, `red_url` (optional `alpha_url`) |
| `opencv.channel.set_alpha` | Set Alpha | `input_url`, `alpha_url` |
| `opencv.channel.channel_mixer` | Channel Mixer | `rr`, `rg`, `rb`, `gr`, `gg`, `gb`, `br`, `bg`, `bb` (mix matrix) |
| `opencv.channel.extract_channel` | Extract Channel | `channel` (R/G/B/A/H/S/V) |
| `opencv.channel.swap_channels` | Swap Channels | `channel_a`, `channel_b` (0/1/2) |

#### segment (5 nodes)

| Node type | Display name | Key inputs |
|---|---|---|
| `opencv.segment.grabcut_segment` | GrabCut Segment | `rect` (x,y,w,h), `iterations` → `mask_url` + `output_url` |
| `opencv.segment.inpaint` | Inpaint | `mask_url`, `radius`, `method` (telea/ns) |
| `opencv.segment.apply_mask` | Apply Mask | `input_url`, `mask_url`, `invert_mask` |
| `opencv.segment.generate_mask_from_color` | Generate Mask from Color | `target_color` (B,G,R), `tolerance` |
| `opencv.segment.flood_fill_mask` | Flood Fill Mask | `seed_x`, `seed_y`, `tolerance` |

#### analysis (2 nodes)

| Node type | Display name | Outputs |
|---|---|---|
| `opencv.analysis.image_statistics` | Image Statistics | `mean_b/g/r`, `std_b/g/r`, `min`, `max`, `width`, `height`, `channels` |
| `opencv.analysis.compare_images` | Compare Images | `psnr`, `mse`, `ssim_approx` |

---

## Project structure

```
├── nodes/
│   ├── registry.py          # Auto-discovery of node modules
│   ├── base.py              # Optional BaseNode class
│   ├── arithmetic/
│   │   ├── add_two.py
│   │   ├── add_three.py
│   │   ├── subtract.py
│   │   ├── multiply.py
│   │   └── divide.py
│   ├── ffmpeg/
│   │   ├── _utils.py        # download_to_tempfile, upload_file, url_suffix
│   │   ├── get_video_info.py
│   │   ├── convert_video.py
│   │   ├── trim_video.py
│   │   ├── resize_video.py
│   │   ├── extract_audio.py
│   │   ├── add_audio.py
│   │   ├── concatenate_videos.py
│   │   ├── extract_frames.py
│   │   ├── video_to_gif.py
│   │   └── add_watermark.py
│   ├── opencv/
│   │   ├── _utils.py        # load_image_from_url, save_and_upload, helpers
│   │   ├── analysis/
│   │   ├── blur/
│   │   ├── channel/
│   │   ├── color/
│   │   ├── composite/
│   │   ├── edge/
│   │   ├── key/
│   │   ├── lens/
│   │   ├── morph/
│   │   ├── noise/
│   │   ├── segment/
│   │   ├── style/
│   │   └── transform/
│   └── core/
│       ├── input_node.py
│       └── output_node.py
├── engine/
│   └── executor.py          # Topological sort + execution
├── api/
│   └── routes.py            # FastAPI route definitions
├── main.py                  # App entry point
└── requirements.txt
```
