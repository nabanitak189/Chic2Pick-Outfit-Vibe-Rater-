# utils/clip_utils.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageOps
import torch
import io
import torch.nn.functional as F

# ---- Load CLIP once (important for speed) ----
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ---- Target vibes (labels we want to classify into) ----
VIBES = ["casual", "formal", "trendy", "sporty", "party"]

# ---- Multiple natural prompts per label (to reduce bias) ----
TEMPLATES = [
    "a photo of a {label} outfit",
    "a full-body street style photo of a {label} outfit",
    "a fashion catalog image showing a {label} look",
    "an instagram photo of a person wearing a {label} outfit",
    "a studio portrait showing {label} clothing",
]


def _load_pil(image_input):
    """
    Accepts different input types and always returns a PIL.Image (RGB).
    Handles: 
      - PIL.Image
      - bytes / bytearray
      - BytesIO (e.g., from st.camera_input)
      - Streamlit UploadedFile
      - file path (str)
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    # Streamlit camera_input returns BytesIO-like object with .getvalue()
    if hasattr(image_input, "getvalue"):
        return Image.open(io.BytesIO(image_input.getvalue())).convert("RGB")

    # Streamlit UploadedFile has .read()
    if hasattr(image_input, "read"):
        raw = image_input.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # File path
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")

    raise ValueError("Unsupported image input type for CLIP.")


def analyze_outfit(image_input):
    """
    Analyze outfit image with CLIP.
    Returns:
        dict {vibe: probability, ... , "_best_label": str, "_best_prob": float}
    """
    # ---- Load and fix orientation (important for camera photos) ----
    img = _load_pil(image_input)
    img = ImageOps.exif_transpose(img)

    # ---- Build expanded text prompt set ----
    prompts = []
    spans = []  # (label_idx, start, end) for averaging templates
    for li, label in enumerate(VIBES):
        start = len(prompts)
        for t in TEMPLATES:
            prompts.append(t.format(label=label))
        end = len(prompts)
        spans.append((li, start, end))

    with torch.no_grad():
        # Encode image
        image_inputs = processor(images=img, return_tensors="pt")
        img_feat = model.get_image_features(**image_inputs)
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

        # Encode all text prompts
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        txt_feat = model.get_text_features(**text_inputs)
        txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity between image and text
        sim = (img_feat @ txt_feat.T)[0]  # shape: [num_prompts]

    # ---- Aggregate similarity per label (average across templates) ----
    label_logits = []
    for li, s, e in spans:
        label_logits.append(sim[s:e].mean())
    label_logits = torch.stack(label_logits)

    # ---- Convert to probabilities ----
    probs = F.softmax(label_logits, dim=0).cpu().numpy()

    # ---- Build output dict ----
    out = {lbl: float(probs[i]) for i, lbl in enumerate(VIBES)}

    # Add best label & probability (always guaranteed)
    best_idx = int(probs.argmax())
    out["_best_label"] = VIBES[best_idx]
    out["_best_prob"] = float(probs[best_idx])

    return out






# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
# import io

# # Load CLIP once (so it doesn't reload every request)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def analyze_outfit(image_input):
#     """
#     Takes an outfit image (file path, bytes, BytesIO, or Streamlit UploadedFile) 
#     and returns vibe scores.
#     """

#     # Define vibes
#     vibes = ["casual", "formal", "trendy", "sporty", "traditional", "party"]

#     # --- Handle different input types safely ---
#     if isinstance(image_input, str):
#         # File path
#         image = Image.open(image_input).convert("RGB")

#     elif isinstance(image_input, bytes):
#         # Raw bytes
#         image = Image.open(io.BytesIO(image_input)).convert("RGB")

#     elif isinstance(image_input, io.BytesIO):
#         # BytesIO object
#         image = Image.open(image_input).convert("RGB")

#     elif hasattr(image_input, "read"):
#         # Streamlit UploadedFile (from uploader or camera)
#         image = Image.open(io.BytesIO(image_input.read())).convert("RGB")

#     else:
#         raise ValueError(f"Unsupported image input type: {type(image_input)}")

#     # --- Process with CLIP ---
#     inputs = processor(text=vibes, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

#     # --- Return dict with rounded probabilities ---
#     return {v: float(round(p, 3)) for v, p in zip(vibes, probs)}
