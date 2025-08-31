
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2  
import numpy as np


def resize_for_analysis(img, max_size=512):
    """
    Resize the image so its largest dimension = max_size (default 512px).
    Keeps aspect ratio intact.
    """
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:  
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def pattern_entropy(img_rgb, mask=None):
    """
    Compute a more robust pattern-density score.
    - Applies a small blur to reduce micro-texture noise (denim threads).
    - Uses a larger entropy disk so only larger patterns register strongly.
    - If a person/mask is provided, computes entropy on the bounding box of the mask
      (focuses on clothing area) and returns a full-size entropy map for debugging.
    Returns:
        score (0..100), entropy_map (uint8 or float)
    """
    try:
        
        gray = rgb2gray(img_rgb)
        
        gray_u8 = img_as_ubyte(cv2.GaussianBlur((gray * 255).astype(np.uint8), (5, 5), 0))
    except Exception:
        
        gray_u8 = img_as_ubyte(rgb2gray(img_rgb))

   
    ent_disk = disk(9)

    if mask is not None:
        
        try:
            if mask.shape != gray_u8.shape:
                mask_resized = cv2.resize(mask, (gray_u8.shape[1], gray_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            mask_bool = (mask_resized > 0)
        except Exception:
            mask_bool = None

        if mask_bool is None or mask_bool.sum() < 100:
            
            ent = entropy(gray_u8, ent_disk)
            ent_map = ent
        else:
            
            ys, xs = np.where(mask_bool)
            y1, y2 = max(0, ys.min()), min(gray_u8.shape[0] - 1, ys.max())
            x1, x2 = max(0, xs.min()), min(gray_u8.shape[1] - 1, xs.max())

            patch = gray_u8[y1:y2+1, x1:x2+1]
            
            if patch.size < 100:
                ent = entropy(gray_u8, ent_disk)
                ent_map = ent
            else:
                ent_patch = entropy(patch, ent_disk)
                
                ent_map = np.zeros_like(gray_u8, dtype=ent_patch.dtype)
                ent_map[y1:y2+1, x1:x2+1] = ent_patch
    else:
        ent_map = entropy(gray_u8, ent_disk)

    
    score = float(np.clip(ent_map.mean() * 6.5, 0, 100))

    return score, ent_map

def colorfulness_score(img_rgb):
    """
    Measures how colorful the image is, using a standard metric (Hasler & Süsstrunk).
    Returns 0–100.
    """
    (B, G, R) = cv2.split(img_rgb.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))

    score = std_root + (0.3 * mean_root)
    return float(np.clip(score, 0, 100))


def edge_complexity(img_rgb):
    """
    Measures visual complexity based on edges (Canny detector).
    Returns 0–100 + edge map for debugging.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    score = 100.0 * (edges > 0).sum() / edges.size
    return float(np.clip(score, 0, 100)), edges


def fit_tightness(img_rgb):
    """
    Estimate fit tightness by segmenting person using GrabCut.
    - Returns tightness score (0..100) and mask (binary).
    """
    h, w = img_rgb.shape[:2]

    
    mask = np.zeros((h, w), np.uint8)

    
    rect = (10, 10, w-20, h-20)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception:
        mask2 = np.ones((h, w), dtype=np.uint8)  

    
    ys, xs = np.where(mask2 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, mask2

    bbox_h = ys.max() - ys.min()
    bbox_w = xs.max() - xs.min()

    
    aspect_ratio = bbox_h / float(bbox_w + 1e-6)
    tightness = np.clip(aspect_ratio * 50, 0, 100)

    return float(tightness), mask2 * 255  


def aggregate_scores(img_rgb):
    """
    Reordered so we compute fit/mask first and then pattern_entropy(masked).
    """
    img_rgb = resize_for_analysis(img_rgb)

   
    color = colorfulness_score(img_rgb)
    complexity, edges = edge_complexity(img_rgb)

    
    fit, mask = fit_tightness(img_rgb)

    
    pattern, ent = pattern_entropy(img_rgb, mask=mask)

   
    minimalist_vs_maximalist = np.clip((complexity * 0.5 + pattern * 0.35 + color * 0.15), 0, 100)
    boldness = np.clip((color * 0.6 + complexity * 0.2 + pattern * 0.2), 0, 100)

    return {
        "colorfulness": float(color),
        "complexity": float(complexity),
        "pattern_density": float(pattern),
        "fit_tightness": float(fit),
        "vibe_maximalism": float(minimalist_vs_maximalist),
        "boldness": float(boldness),
        "debug": {
            "edges": edges,
            "entropy": ent,
            "mask": mask
        },
        "processed_rgb": img_rgb
    }