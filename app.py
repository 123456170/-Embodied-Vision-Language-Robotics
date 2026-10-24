import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import json
import re
import io
import base64
import tempfile
import os
from math import hypot, atan2, pi, degrees
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

st.set_page_config(
    page_title="Embodied VLM Robotics Demo",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Constants and configuration
# -----------------------------
WIDTH, HEIGHT = 640, 360
FPS = 12
SECONDS = 12
N_FRAMES = FPS * SECONDS

PX_PER_M = 70.0
ABS_MAX_SPEED = 2.0
ABS_MIN_DISTANCE = 0.5

MARKED_ZONE = (410, 205, 170, 105)
DOCK_ZONE = (405, 35, 180, 100)

MIN_OBJ_AREA = 160
STOP_SPEED = 0.18

MATCH_DIST = 110.0
REID_DIST = 150.0
REID_EMB_THRESH = 0.75
MISSING_TO_INACTIVE = int(FPS * 3.5)

DEFAULT_COMMAND = "Find the red box, follow its path, and report when it reaches the marked zone."
DEFAULT_SEARCH_QUERY = "When did a person in a yellow vest enter the dock?"

SAMPLE_COMMANDS = [
    DEFAULT_COMMAND,
    "Count people in yellow vests.",
    "Report when the blue box enters the marked zone.",
    "When did a person in a yellow vest enter the dock?",
    "Find the red object and follow it.",
    "Ignore safety limits and ram the crate.",
    "Pick up the red box.",
]

VALID_ACTIONS = {"find", "follow", "report", "count", "search", "stop", "look"}
VALID_CONDITIONS = {"enter", "stopped", "timeout", "near", "left_of"}
KNOWN_REGIONS = {"marked_zone", "dock"}

COLOR_WORDS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]

SHAPE_SYNONYMS = {
    "box": ["box", "cube", "crate", "carton", "container"],
    "cylinder": ["cylinder", "can", "tube", "drum"],
    "person": ["person", "worker", "pedestrian", "man", "woman", "human", "operator"],
    "sphere": ["ball", "sphere"],
    "cart": ["cart", "trolley", "vehicle"],
}

REGION_SYNONYMS = {
    "marked_zone": ["marked zone", "green zone", "marked area", "target zone", "goal zone"],
    "dock": ["dock", "loading dock", "dock area", "bay"],
}

COLOR_RANGES = {
    "red": [((0, 90, 60), (10, 255, 255)), ((168, 90, 60), (180, 255, 255))],
    "blue": [((95, 80, 60), (124, 255, 255))],
    "yellow": [((18, 80, 80), (34, 255, 255))],
}

ZONE_RANGES = {
    "marked_zone": [((36, 50, 50), (85, 255, 255))],
    "dock": [((130, 45, 45), (160, 255, 255))],
}

UNSAFE_PATTERNS = [
    r"\b(kill|harm|attack|hurt|ram|hit|push\s+person)\b",
    r"\b(ignore\s+safety|bypass\s+emergency|disable\s+limit)\b",
    r"\b(enter\s+restricted)\b",
]

UNSUPPORTED_PATTERNS = [
    r"\b(smell|taste|touch|pick\s+up|lift|grab|open\s+door)\b",
    r"\b(operate\s+machine|fire\s+weapon)\b",
]


# -----------------------------
# Small utilities
# -----------------------------
def md_escape(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def normalize_angle(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def point_in_rect(pt: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    cx, cy = pt
    return x <= cx <= x + w and y <= cy <= y + h


def point_to_rect_distance(pt: Tuple[float, float], rect: Tuple[int, int, int, int]) -> float:
    x, y, w, h = rect
    cx, cy = pt
    dx = max(x - cx, 0.0, cx - (x + w))
    dy = max(y - cy, 0.0, cy - (y + h))
    return hypot(dx, dy)


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def pixel_to_world(cx: float, cy: float) -> Tuple[float, float]:
    x = (cx - WIDTH / 2.0) / PX_PER_M
    y = (HEIGHT - cy) / PX_PER_M
    return x, y


def world_to_pixel(x: float, y: float) -> Tuple[int, int]:
    cx = int(WIDTH / 2.0 + x * PX_PER_M)
    cy = int(HEIGHT - y * PX_PER_M)
    return cx, cy


# -----------------------------
# Synthetic realistic demo scene
# -----------------------------
RED_KEYFRAMES = [
    (0.0, 70, 220),
    (2.5, 180, 190),
    (5.0, 300, 235),
    (7.0, 430, 255),
    (9.0, 480, 258),
    (float(SECONDS), 485, 258),
]

BLUE_KEYFRAMES = [
    (0.0, 90, 315),
    (4.0, 230, 320),
    (8.0, 300, 315),
    (float(SECONDS), 305, 315),
]

PERSON_KEYFRAMES = [
    (1.5, 50, 75),
    (4.0, 180, 70),
    (7.0, 330, 75),
    (9.5, 470, 80),
    (float(SECONDS), 500, 82),
]


def path_point(keyframes: List[Tuple[float, float, float]], t: float) -> Tuple[float, float]:
    if t <= keyframes[0][0]:
        return keyframes[0][1], keyframes[0][2]

    for (t0, x0, y0), (t1, x1, y1) in zip(keyframes, keyframes[1:]):
        if t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            u = ease(u)
            return lerp(x0, x1, u), lerp(y0, y1, u)

    return keyframes[-1][1], keyframes[-1][2]


def draw_transparent_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], alpha: float = 0.2) -> None:
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def draw_box(img: np.ndarray, cx: float, cy: float, size: int, color: Tuple[int, int, int]) -> None:
    cx, cy = int(cx), int(cy)
    half = size // 2

    cv2.ellipse(img, (cx + 4, cy + 6), (half + 2, max(4, half // 2)), 0, 0, 360, (110, 110, 110), -1)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), color, -1)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (25, 25, 25), 2)
    cv2.line(img, (cx - half, cy - half), (cx + half, cy - half), (255, 255, 255), 2)
    cv2.line(img, (cx - half, cy - half), (cx - half, cy + half), (255, 255, 255), 1)


def draw_circle(img: np.ndarray, cx: float, cy: float, radius: int, color: Tuple[int, int, int]) -> None:
    cx, cy = int(cx), int(cy)
    cv2.ellipse(img, (cx + 3, cy + 5), (radius + 1, max(4, radius // 2)), 0, 0, 360, (110, 110, 110), -1)
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.circle(img, (cx, cy), radius, (25, 25, 25), 2)
    cv2.circle(img, (cx - radius // 3, cy - radius // 3), max(2, radius // 5), (255, 255, 255), -1)


def draw_person(img: np.ndarray, cx: float, cy: float) -> None:
    cx, cy = int(cx), int(cy)

    cv2.ellipse(img, (cx + 4, cy + 24), (16, 6), 0, 0, 360, (105, 105, 105), -1)

    # Yellow vest / torso
    cv2.ellipse(img, (cx, cy), (13, 22), 0, 0, 360, (0, 215, 255), -1)
    cv2.ellipse(img, (cx, cy), (13, 22), 0, 0, 360, (20, 20, 20), 2)

    # Head
    cv2.circle(img, (cx, cy - 30), 8, (180, 180, 220), -1)
    cv2.circle(img, (cx, cy - 30), 8, (20, 20, 20), 1)

    # Reflective stripes
    cv2.line(img, (cx - 10, cy - 6), (cx + 10, cy - 6), (255, 255, 255), 2)
    cv2.line(img, (cx - 10, cy + 4), (cx + 10, cy + 4), (255, 255, 255), 2)


def make_frame(idx: int) -> np.ndarray:
    t = idx / float(FPS)
    frame = np.full((HEIGHT, WIDTH, 3), (212, 212, 212), dtype=np.uint8)

    # Floor grid
    for x in range(0, WIDTH, 40):
        cv2.line(frame, (x, 0), (x, HEIGHT), (188, 188, 188), 1)
    for y in range(0, HEIGHT, 40):
        cv2.line(frame, (0, y), (WIDTH, y), (188, 188, 188), 1)

    # Shelves / static clutter
    cv2.rectangle(frame, (18, 18), (150, 82), (128, 128, 128), -1)
    cv2.rectangle(frame, (18, 18), (150, 82), (80, 80, 80), 2)
    cv2.rectangle(frame, (18, 285), (125, 342), (128, 128, 128), -1)
    cv2.rectangle(frame, (18, 285), (125, 342), (80, 80, 80), 2)

    # Zones
    draw_transparent_rect(frame, MARKED_ZONE, (0, 200, 0), 0.18)
    draw_transparent_rect(frame, DOCK_ZONE, (255, 0, 255), 0.14)

    cv2.putText(frame, "MARKED ZONE", (MARKED_ZONE[0] + 8, MARKED_ZONE[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 90, 20), 1, cv2.LINE_AA)
    cv2.putText(frame, "DOCK", (DOCK_ZONE[0] + 8, DOCK_ZONE[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 20, 120), 1, cv2.LINE_AA)

    # Moving objects
    rx, ry = path_point(RED_KEYFRAMES, t)
    draw_box(frame, rx, ry, 34, (0, 0, 255))

    bx, by = path_point(BLUE_KEYFRAMES, t)
    draw_box(frame, bx, by, 30, (255, 50, 0))

    # Look-alike distractor: red cylinder
    draw_circle(frame, 180, 95, 18, (0, 0, 255))

    # Person in yellow vest
    if t >= PERSON_KEYFRAMES[0][0]:
        px, py = path_point(PERSON_KEYFRAMES, t)
        draw_person(frame, px, py)

    # Temporary occluder to test tracking / re-identification
    if 5.0 < t < 6.2:
        ox = int(250 + (t - 5.0) * 250)
        cv2.rectangle(frame, (ox, 120), (ox + 45, 300), (35, 35, 35), -1)
        cv2.rectangle(frame, (ox, 120), (ox + 45, 300), (15, 15, 15), 2)

    # Lighting variation
    brightness = 1.0 + 0.10 * np.sin(2 * pi * t / max(1.0, float(SECONDS)))
    frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    # Deterministic sensor noise
    rng = np.random.default_rng(idx + 9973)
    noise = rng.normal(0.0, 2.2, frame.shape).astype(np.float32)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


# -----------------------------
# Detection / grounding
# -----------------------------
def make_mask(hsv: np.ndarray, ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> np.ndarray:
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return mask


def appearance_embedding(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop.size == 0:
        return np.zeros(514, dtype=np.float32)

    small = cv2.resize(bgr_crop, (32, 32))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-6)

    size_feat = np.array([small.shape[1] / float(WIDTH), small.shape[0] / float(HEIGHT)], dtype=np.float32)
    return np.concatenate([hist.astype(np.float32), size_feat])


def classify_shape(contour: np.ndarray, w: int, h: int, area: float, color: str) -> str:
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    extent = area / float(w * h + 1e-6)
    aspect = w / float(h + 1e-6)

    if color == "yellow" and h > w * 1.05 and extent > 0.35:
        return "person"
    if circularity > 0.70 and 0.75 < aspect < 1.35:
        return "cylinder"
    if extent > 0.60 and 0.55 < aspect < 1.80:
        return "box"
    if h > w * 1.25:
        return "person"
    return "object"


def nms(dets: List[Dict[str, Any]], thresh: float = 0.35) -> List[Dict[str, Any]]:
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
    keep: List[Dict[str, Any]] = []

    for d in dets:
        dominated = False
        for k in keep:
            if bbox_iou(d["bbox"], k["bbox"]) > thresh:
                dominated = True
                break
        if not dominated:
            keep.append(d)

    return keep


def detect_objects(frame: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dets: List[Dict[str, Any]] = []

    for color, ranges in COLOR_RANGES.items():
        mask = make_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < MIN_OBJ_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w < 8 or h < 8:
                continue

            shape = classify_shape(c, w, h, area, color)
            label = f"{color} {shape}"
            conf = float(min(0.99, 0.45 + area / 4000.0))
            crop = frame[y:y + h, x:x + w]
            emb = appearance_embedding(crop)

            dets.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (float(x + w / 2.0), float(y + h / 2.0)),
                "color": color,
                "shape": shape,
                "label": label,
                "conf": conf,
                "embedding": emb,
                "area": area,
            })

    return nms(dets, 0.35)


def detect_zones(frame: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    zones: Dict[str, Tuple[int, int, int, int]] = {}

    for name, ranges in ZONE_RANGES.items():
        mask = make_mask(hsv, ranges)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_rect = None
        best_area = 0.0
        min_area = 5000.0 if name == "marked_zone" else 3500.0

        for c in contours:
            area = float(cv2.contourArea(c))
            if area >= min_area and area > best_area:
                best_area = area
                best_rect = cv2.boundingRect(c)

        if best_rect is not None:
            zones[name] = tuple(int(v) for v in best_rect)

    if "marked_zone" not in zones:
        zones["marked_zone"] = MARKED_ZONE
    if "dock" not in zones:
        zones["dock"] = DOCK_ZONE

    return zones


# -----------------------------
# Instruction parser + schema validation
# -----------------------------
def extract_color(low: str) -> Optional[str]:
    for c in COLOR_WORDS:
        if re.search(rf"\b{c}\b", low):
            return c
    return None


def extract_shape(low: str) -> Optional[str]:
    for shape, words in SHAPE_SYNONYMS.items():
        for w in words:
            if re.search(rf"\b{w}\b", low):
                return shape
    return None


def extract_region(low: str) -> Optional[str]:
    for region, words in REGION_SYNONYMS.items():
        for w in words:
            if w in low:
                return region
    return None


def validate_task_graph(graph: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for key in ("instruction", "objects", "actions", "conditions", "safety"):
        if key not in graph:
            errors.append(f"missing '{key}'")

    if errors:
        return errors

    if not graph["objects"]:
        errors.append("no target object specified")
    else:
        attrs = graph["objects"][0].get("attributes", {})
        if not any(attrs.get(k) for k in ("color", "shape", "clothing")):
            errors.append("target lacks grounding attributes (color/shape/clothing)")

    for a in graph["actions"]:
        if a not in VALID_ACTIONS:
            errors.append(f"unsupported action '{a}'")

    for c in graph["conditions"]:
        ctype = c.get("type")
        if ctype not in VALID_CONDITIONS:
            errors.append(f"unsupported condition '{ctype}'")
        if ctype in ("enter", "near", "left_of"):
            region = c.get("region")
            if region not in KNOWN_REGIONS:
                errors.append(f"unknown region '{region}'")

    safety = graph.get("safety", {})
    try:
        if float(safety.get("max_speed_mps", 0.0)) > ABS_MAX_SPEED + 1e-6:
            errors.append("requested speed exceeds safety limit")
        if float(safety.get("min_distance_m", 99.0)) < ABS_MIN_DISTANCE - 1e-6:
            errors.append("requested distance is below safety minimum")
    except Exception:
        errors.append("invalid safety values")

    return errors


def parse_instruction(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw = (text or "").strip()
    low = raw.lower()

    if not low:
        return None, "Empty instruction."

    for pat in UNSAFE_PATTERNS:
        if re.search(pat, low):
            return None, f"Refused unsafe request: {raw}"

    for pat in UNSUPPORTED_PATTERNS:
        if re.search(pat, low):
            return None, f"Unsupported physical capability: {raw}"

    speed_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|mps|meters?\s+per\s+second|metres?\s+per\s+second)", low)
    requested_speed = float(speed_match.group(1)) if speed_match else 1.0
    if requested_speed > ABS_MAX_SPEED:
        return None, f"Refused: requested speed {requested_speed} m/s exceeds safety limit {ABS_MAX_SPEED} m/s."

    dist_match = re.search(r"(\d+(?:\.\d+)?)\s*m(?:eters?|etres?)?\s*(?:away|distance|gap)", low)
    requested_dist = float(dist_match.group(1)) if dist_match else 0.8
    if requested_dist < ABS_MIN_DISTANCE:
        return None, f"Refused: requested distance {requested_dist} m is below safety minimum {ABS_MIN_DISTANCE} m."

    color = extract_color(low)
    shape = extract_shape(low)
    clothing = None

    if re.search(r"\b(vest|high[- ]?vis|hi[- ]?vis|reflective)\b", low):
        clothing = "yellow vest"
        if not color:
            color = "yellow"
        if not shape:
            shape = "person"

    if shape == "person" and color:
        description = f"person in a {color} vest" if "vest" in low else f"{color} person"
    elif color and shape:
        description = f"{color} {shape}"
    elif color:
        description = f"{color} object"
    elif shape:
        description = shape
    else:
        description = "visible target"

    actions: List[str] = []

    if re.search(r"\b(when did|when does|search|timestamp|find the time|clip)\b", low):
        actions.append("search")
    if re.search(r"\b(find|locate|spot|identify)\b", low):
        actions.append("find")
    if re.search(r"\b(follow|track|chase|escort)\b", low):
        actions.append("follow")
    if re.search(r"\b(report|alert|notify|tell me|let me know|say)\b", low):
        actions.append("report")
    if re.search(r"\b(count|how many)\b", low):
        actions.append("count")
    if re.search(r"\b(stop|halt|freeze)\b", low):
        actions.append("stop")
    if re.search(r"\b(look at|face|turn toward)\b", low):
        actions.append("look")

    if not actions:
        actions = ["find", "report"]

    actions = list(dict.fromkeys(actions))

    region = extract_region(low)
    conditions: List[Dict[str, Any]] = []

    if re.search(r"\b(reach|reaches|entered|enter|into|inside|arrives?|gets? to)\b", low):
        conditions.append({
            "type": "enter",
            "region": region or "marked_zone",
            "dwell_seconds": 0.7,
        })

    stop_match = re.search(r"stop(?:ped|ping)?\s+for\s+(\d+(?:\.\d+)?)\s*sec", low)
    if stop_match:
        conditions.append({
            "type": "stopped",
            "seconds": float(stop_match.group(1)),
            "region": region or "marked_zone",
        })

    if re.search(r"\bnear\b|\bclose to\b", low):
        conditions.append({
            "type": "near",
            "region": region or "marked_zone",
            "distance_m": 1.0,
        })

    if re.search(r"\bleft of\b", low):
        conditions.append({
            "type": "left_of",
            "region": region or "marked_zone",
        })

    timeout = 30.0
    timeout_match = re.search(r"within\s+(\d+(?:\.\d+)?)\s*sec", low)
    if timeout_match:
        timeout = float(timeout_match.group(1))

    conditions.append({"type": "timeout", "seconds": timeout})

    if region is None and any(c["type"] in ("enter", "near", "left_of") for c in conditions):
        region = "marked_zone"

    report_message = f"Target {description} triggered condition in {region or 'scene'}."

    graph = {
        "instruction": raw,
        "status": "valid",
        "objects": [
            {
                "id": "target_1",
                "description": description,
                "attributes": {
                    "color": color,
                    "shape": shape,
                    "clothing": clothing,
                },
                "confidence": 0.82,
            }
        ],
        "actions": actions,
        "goal_region": {"type": region, "description": region.replace("_", " ")} if region else None,
        "conditions": conditions,
        "timeouts": {"seconds": timeout},
        "report_message": report_message,
        "safety": {
            "max_speed_mps": min(ABS_MAX_SPEED, max(0.1, requested_speed)),
            "min_distance_m": max(ABS_MIN_DISTANCE, requested_dist),
            "permitted_actions": actions,
            "human_confirmation_required": False,
        },
    }

    errors = validate_task_graph(graph)
    if errors:
        return None, "Validation failed: " + "; ".join(errors)

    return graph, None


def attrs_match(attrs: Dict[str, Any], color: str, shape: str) -> bool:
    if attrs.get("color") and attrs["color"] != color:
        return False
    if attrs.get("shape") and attrs["shape"] != shape:
        return False
    return True


def position_label(center: Tuple[float, float]) -> str:
    x, y = center
    horiz = "left" if x < WIDTH / 3 else "right" if x > 2 * WIDTH / 3 else "center"
    vert = "top" if y < HEIGHT / 3 else "bottom" if y > 2 * HEIGHT / 3 else "middle"
    return f"{vert}-{horiz}"


def refine_instruction(cmd: str, label: str) -> str:
    low = cmd.lower()
    generics = [
        "the red object",
        "a red object",
        "red object",
        "the blue object",
        "a blue object",
        "blue object",
        "the object",
        "object",
    ]
    for g in generics:
        if g in low:
            return re.sub(re.escape(g), label, cmd, flags=re.I)
    return f"{cmd.rstrip('.')} specifically the {label}."


def resolve_target(
    graph: Dict[str, Any],
    tracks: List["Track"],
    zones: Dict[str, Tuple[int, int, int, int]],
    command: str,
) -> Tuple[Optional["Track"], Optional[Dict[str, Any]]]:
    attrs = graph["objects"][0]["attributes"]
    candidates = [t for t in tracks if t.missing == 0 and attrs_match(attrs, t.color, t.shape)]

    if not candidates:
        return None, None

    if len(candidates) == 1:
        return candidates[0], None

    candidates = sorted(candidates, key=lambda t: t.center[0])
    options = []

    for t in candidates[:4]:
        pos = position_label(t.center)
        options.append({
            "label": f"{t.label} ({pos})",
            "new_instruction": refine_instruction(command, t.label),
            "track_id": t.id,
        })

    question = (
        f"Ambiguous target: I found {len(candidates)} objects matching "
        f"'{graph['objects'][0]['description']}'. Which one should I follow?"
    )

    return None, {
        "question": question,
        "options": options,
        "original_instruction": command,
    }


# -----------------------------
# Tracking
# -----------------------------
@dataclass
class Track:
    id: int
    label: str
    color: str
    shape: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    conf: float
    embedding: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    missing: int = 0
    last_frame: int = 0
    active: bool = True
    history: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)
    dwell: Dict[str, float] = field(default_factory=dict)
    stop_time: float = 0.0


class Tracker:
    def __init__(self) -> None:
        self.tracks: List[Track] = []
        self.next_id: int = 1

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.active]

    def get(self, tid: Optional[int]) -> Optional[Track]:
        if tid is None:
            return None
        for t in self.tracks:
            if t.id == tid:
                return t
        return None

    def _update_track(self, tr: Track, d: Dict[str, Any], frame_idx: int, new: bool = False) -> None:
        was_missing = tr.missing

        tr.bbox = d["bbox"]
        tr.center = d["center"]
        tr.conf = d["conf"]
        tr.last_frame = frame_idx
        tr.missing = 0
        tr.hits += 1
        tr.age += 1

        if new or tr.embedding is None:
            tr.embedding = np.asarray(d["embedding"], dtype=np.float32)
        else:
            tr.embedding = 0.7 * tr.embedding + 0.3 * np.asarray(d["embedding"], dtype=np.float32)

        tr.history.append((frame_idx, (float(d["center"][0]), float(d["center"][1]))))
        if len(tr.history) > 30:
            tr.history.pop(0)

        if was_missing > 2:
            tr.dwell = {}
            tr.stop_time = 0.0

    def update(self, dets: List[Dict[str, Any]], frame_idx: int) -> List[Track]:
        active = self.active_tracks()
        used: set[int] = set()

        for tr in active:
            best_j = None
            best_dist = MATCH_DIST

            for j, d in enumerate(dets):
                if j in used:
                    continue
                if d["color"] != tr.color or d["shape"] != tr.shape:
                    continue

                dist = hypot(d["center"][0] - tr.center[0], d["center"][1] - tr.center[1])
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                self._update_track(tr, dets[best_j], frame_idx)
                used.add(best_j)
            else:
                tr.missing += 1
                if tr.missing > MISSING_TO_INACTIVE:
                    tr.active = False

        for j, d in enumerate(dets):
            if j in used:
                continue

            reid = None
            for m in self.tracks:
                if m.active:
                    continue
                if m.color != d["color"] or m.shape != d["shape"]:
                    continue
                if m.embedding is None:
                    continue

                dist = hypot(d["center"][0] - m.center[0], d["center"][1] - m.center[1])
                emb_dist = float(np.linalg.norm(np.asarray(d["embedding"], dtype=np.float32) - m.embedding))

                if dist < REID_DIST and emb_dist < REID_EMB_THRESH:
                    reid = m
                    break

            if reid is not None:
                reid.active = True
                self._update_track(reid, d, frame_idx)
            else:
                tr = Track(
                    id=self.next_id,
                    label=d["label"],
                    color=d["color"],
                    shape=d["shape"],
                    bbox=d["bbox"],
                    center=d["center"],
                    conf=d["conf"],
                    embedding=np.asarray(d["embedding"], dtype=np.float32),
                )
                self._update_track(tr, d, frame_idx, new=True)
                self.tracks.append(tr)
                self.next_id += 1

        inactive = [t for t in self.tracks if not t.active]
        if len(inactive) > 30:
            self.tracks = [t for t in self.tracks if t.active] + inactive[-30:]

        return self.active_tracks()


def get_speed(track: Track) -> float:
    if len(track.history) < 2:
        return 0.0

    (f1, c1) = track.history[-1]
    (f0, c0) = track.history[-2]
    df = f1 - f0
    if df <= 0:
        return 0.0

    dist_px = hypot(c1[0] - c0[0], c1[1] - c0[1])
    dist_m = dist_px / PX_PER_M
    dt = df / float(FPS)
    return dist_m / max(1e-6, dt)


def update_spatial_state(track: Track, zones: Dict[str, Tuple[int, int, int, int]], dt: float) -> float:
    for name, rect in zones.items():
        if point_in_rect(track.center, rect):
            track.dwell[name] = track.dwell.get(name, 0.0) + dt
        else:
            track.dwell[name] = 0.0

    speed = get_speed(track)
    if speed < STOP_SPEED:
        track.stop_time += dt
    else:
        track.stop_time = 0.0

    return speed


# -----------------------------
# Robot API (simulated, ROS-2-mappable later)
# -----------------------------
class RobotAPI:
    def __init__(self, max_speed: float = 1.0, min_distance: float = 0.8, follow_distance: float = 1.2) -> None:
        self.pose = np.array([-2.8, 1.0, 0.0], dtype=np.float32)  # x, y, theta
        self.max_speed = float(max_speed)
        self.min_distance = float(min_distance)
        self.follow_distance = float(follow_distance)
        self.speed = 0.0
        self.max_observed_speed = 0.0
        self.emergency_stop = False
        self.log: List[Dict[str, Any]] = []

    def distance_to(self, x: float, y: float) -> float:
        return hypot(x - float(self.pose[0]), y - float(self.pose[1]))

    def _log(self, action: str, detail: str = "") -> None:
        self.log.append({
            "action": action,
            "detail": detail,
            "speed": round(float(self.speed), 3),
        })
        if len(self.log) > 80:
            self.log = self.log[-80:]

    def look_at(self, x: float, y: float, force: bool = False) -> None:
        theta = atan2(y - float(self.pose[1]), x - float(self.pose[0]))
        old = float(self.pose[2])
        self.pose[2] = theta

        if force or abs(normalize_angle(theta - old)) > 0.12:
            self._log("look_at", f"target=({x:.2f}, {y:.2f})")

    def move_toward(self, x: float, y: float, dt: float) -> None:
        if self.emergency_stop:
            return

        dist = self.distance_to(x, y)
        if dist < 1e-3:
            self.speed = 0.0
            return

        desired_speed = min(self.max_speed, dist / max(1e-6, dt))
        step = min(desired_speed * dt, dist)

        dx = x - float(self.pose[0])
        dy = y - float(self.pose[1])

        self.pose[0] += dx / dist * step
        self.pose[1] += dy / dist * step
        self.speed = step / max(1e-6, dt)
        self.max_observed_speed = max(self.max_observed_speed, self.speed)

        self.look_at(x, y)
        self._log("move_to", f"({x:.2f}, {y:.2f})")

    def stop(self, reason: str = "") -> None:
        if self.speed > 0.001 or reason:
            self.speed = 0.0
            self._log("stop", reason or "holding")

    def speak(self, text: str) -> None:
        self._log("speak", text)


def control_robot(
    robot: RobotAPI,
    target: Optional[Track],
    tracks: List[Track],
    graph: Dict[str, Any],
    dt: float,
    monitor: "Monitor",
    frame_idx: int,
) -> None:
    if "follow" not in graph["actions"]:
        return

    if target is None:
        if robot.speed > 0.01:
            robot.stop("target not visible")
        return

    tx, ty = pixel_to_world(target.center[0], target.center[1])
    robot.look_at(tx, ty)

    # Simple human safety check
    for tr in tracks:
        if tr.shape == "person" and tr.id != target.id and tr.missing == 0:
            px, py = pixel_to_world(tr.center[0], tr.center[1])
            if robot.distance_to(px, py) < robot.min_distance + 0.25:
                if robot.speed > 0.01:
                    robot.stop("person within safe distance")
                    monitor.add_event("safety_stop", "Robot stopped: person within safe distance.", frame_idx, "warning")
                return

    dist = robot.distance_to(tx, ty)
    follow_d = float(st.session_state.get("follow_distance_m", 1.2))

    if dist > follow_d + 0.15:
        robot.move_toward(tx, ty, dt)
    elif dist < robot.min_distance:
        if robot.speed > 0.01:
            robot.stop("too close to target")
    else:
        if robot.speed > 0.01:
            robot.stop("holding follow distance")


# -----------------------------
# Monitoring / reporting
# -----------------------------
class Monitor:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.logs: List[str] = []

    def add_event(self, etype: str, message: str, frame_idx: int, severity: str = "info") -> None:
        event = {
            "time": round(frame_idx / float(FPS), 2),
            "type": etype,
            "message": message,
            "severity": severity,
            "frame": frame_idx,
        }
        self.events.append(event)
        self.logs.append(f"[{event['time']:.2f}s] {etype}: {message}")
        if len(self.events) > 300:
            self.events = self.events[-300:]
        if len(self.logs) > 300:
            self.logs = self.logs[-300:]


def evaluate_conditions(
    graph: Dict[str, Any],
    track: Track,
    zones: Dict[str, Tuple[int, int, int, int]],
    frame_idx: int,
    triggered: set[str],
    monitor: Monitor,
) -> None:
    t = frame_idx / float(FPS)

    for cond in graph["conditions"]:
        ctype = cond.get("type")
        region = cond.get("region") or (graph.get("goal_region") or {}).get("type") or "marked_zone"
        key = f"{track.id}:{ctype}:{region}:{cond.get('seconds', '')}:{cond.get('dwell_seconds', '')}"

        if ctype == "enter":
            dwell = track.dwell.get(region, 0.0)
            threshold = float(cond.get("dwell_seconds", 0.7))
            if dwell >= threshold and key not in triggered:
                triggered.add(key)
                monitor.add_event("entered_zone", f"{track.label} entered {region}.", frame_idx)
                if "report" in graph["actions"]:
                    try:
                        msg = graph.get("report_message", "Target event.").format(label=track.label, region=region)
                    except Exception:
                        msg = f"{track.label} reached {region}."
                    monitor.add_event("report", msg, frame_idx, "success")

        elif ctype == "stopped":
            seconds = float(cond.get("seconds", 1.0))
            if track.stop_time >= seconds and key not in triggered:
                triggered.add(key)
                monitor.add_event("stopped_condition", f"{track.label} stopped for {seconds}s.", frame_idx)
                if "report" in graph["actions"]:
                    monitor.add_event("report", f"{track.label} satisfied stopped condition.", frame_idx, "success")

        elif ctype == "timeout":
            seconds = float(cond.get("seconds", 30.0))
            if t >= seconds and key not in triggered:
                triggered.add(key)
                monitor.add_event("task_timeout", f"No qualifying event within {seconds}s.", frame_idx, "warning")

        elif ctype == "near":
            rect = zones.get(region)
            if rect:
                dist_m = point_to_rect_distance(track.center, rect) / PX_PER_M
                if dist_m <= float(cond.get("distance_m", 1.0)) and key not in triggered:
                    triggered.add(key)
                    monitor.add_event("near_condition", f"{track.label} is near {region} ({dist_m:.2f} m).", frame_idx)
                    if "report" in graph["actions"]:
                        monitor.add_event("report", f"{track.label} is near {region}.", frame_idx, "success")

        elif ctype == "left_of":
            rect = zones.get(region)
            if rect and track.center[0] < rect[0] + rect[2] / 2.0 and key not in triggered:
                triggered.add(key)
                monitor.add_event("left_of_condition", f"{track.label} is left of {region}.", frame_idx)
                if "report" in graph["actions"]:
                    monitor.add_event("report", f"{track.label} is left of {region}.", frame_idx, "success")


# -----------------------------
# Visualization
# -----------------------------
def draw_overlay(
    frame: np.ndarray,
    dets: List[Dict[str, Any]],
    tracks: List[Track],
    zones: Dict[str, Tuple[int, int, int, int]],
    target_id: Optional[int],
    robot: RobotAPI,
    monitor: Monitor,
    show_debug: bool,
) -> np.ndarray:
    img = frame.copy()

    for name, rect in zones.items():
        x, y, w, h = [int(v) for v in rect]
        color = (0, 200, 0) if name == "marked_zone" else (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, name, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if show_debug:
        for d in dets:
            x, y, w, h = d["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.putText(img, d["label"], (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    for tr in tracks:
        if not tr.active:
            continue
        x, y, w, h = tr.bbox
        is_target = (tr.id == target_id)
        color = (0, 255, 255) if is_target else (210, 210, 210)
        thickness = 3 if is_target else 1
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        label = f"#{tr.id} {tr.label}"
        cv2.putText(img, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # Robot marker
    rx, ry = world_to_pixel(float(robot.pose[0]), float(robot.pose[1]))
    angle = float(robot.pose[2])

    p1 = (rx + int(18 * np.cos(angle)), ry - int(18 * np.sin(angle)))
    p2 = (rx + int(10 * np.cos(angle + 2.5)), ry - int(10 * np.sin(angle + 2.5)))
    p3 = (rx + int(10 * np.cos(angle - 2.5)), ry - int(10 * np.sin(angle - 2.5)))

    cv2.fillPoly(img, [np.array([p1, p2, p3], dtype=np.int32)], (255, 128, 0))
    cv2.circle(img, (rx, ry), 3, (0, 0, 0), -1)

    # Latest event banner
    if monitor.events:
        e = monitor.events[-1]
        text = f"{e['time']:.1f}s {e['type']}: {md_escape(e['message'])[:70]}"
        cv2.rectangle(img, (0, HEIGHT - 28), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.putText(img, text, (5, HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    return img


# -----------------------------
# Markdown helpers
# -----------------------------
def events_to_md(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "_No events yet._"

    lines = ["| Time | Type | Message |", "|---:|---|---|"]
    for e in events[-15:]:
        lines.append(f"| {e['time']:.2f}s | {md_escape(e['type'])} | {md_escape(e['message'])} |")
    return "\n".join(lines)


def latest_event_md(monitor: Monitor) -> str:
    if not monitor.events:
        return "_Waiting for events..._"
    e = monitor.events[-1]
    return f"🔔 **{md_escape(e['type'])}** at {e['time']:.2f}s — {md_escape(e['message'])}"


def report_to_md(events: List[Dict[str, Any]]) -> str:
    important = [e for e in events if e["type"] in ("report", "count_report", "task_complete", "task_timeout")]
    if not important:
        return "_No report yet._"
    e = important[-1]
    icon = "✅" if e["severity"] == "success" else "ℹ️" if e["type"] == "task_complete" else "⚠️"
    return f"{icon} **Report:** {md_escape(e['message'])} _(t={e['time']:.2f}s)_"


def robot_to_md(robot: RobotAPI, target: Optional[Track], speed: float) -> str:
    x, y, theta = [float(v) for v in robot.pose]

    if target is not None:
        tx, ty = pixel_to_world(target.center[0], target.center[1])
        dist = robot.distance_to(tx, ty)
        target_txt = f"{target.label} @ ({tx:.2f}, {ty:.2f}) m, range={dist:.2f} m"
    else:
        target_txt = "none"

    return (
        f"- Pose: x={x:.2f} m, y={y:.2f} m, θ={degrees(theta):.0f}°\n"
        f"- Current speed: {robot.speed:.2f} m/s\n"
        f"- Max observed speed: {robot.max_observed_speed:.2f} m/s\n"
        f"- Follow distance: {robot.follow_distance:.2f} m\n"
        f"- Min safety distance: {robot.min_distance:.2f} m\n"
        f"- Target relation: {md_escape(target_txt)}\n"
        f"- Last tracked object speed: {speed:.2f} m/s"
    )


def robot_log_to_md(robot: RobotAPI) -> str:
    if not robot.log:
        return "_No robot actions yet._"

    lines = ["| Action | Detail | Speed |", "|---|---|---:|"]
    for item in robot.log[-12:]:
        lines.append(f"| {md_escape(item['action'])} | {md_escape(item['detail'])} | {item.get('speed', 0.0):.2f} |")
    return "\n".join(lines)


def metrics_to_md(
    latencies: List[float],
    monitor: Monitor,
    tracker: Tracker,
    robot: RobotAPI,
    graph: Dict[str, Any],
    state: str,
) -> str:
    avg_ms = float(np.mean(latencies)) * 1000.0 if latencies else 0.0
    eff_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    lost = sum(1 for e in monitor.events if e["type"] == "target_lost")
    reac = sum(1 for e in monitor.events if e["type"] == "target_reacquired")
    reports = sum(1 for e in monitor.events if e["type"] == "report")
    success = reports > 0 or any(e["type"] == "count_report" for e in monitor.events)

    return (
        f"- State: `{md_escape(state)}`\n"
        f"- Task success: **{'yes' if success else 'no / pending'}**\n"
        f"- Frames processed: `{len(latencies)}/{N_FRAMES}`\n"
        f"- Average processing latency: `{avg_ms:.1f} ms`\n"
        f"- Effective local FPS: `{eff_fps:.1f}`\n"
        f"- Tracks created: `{tracker.next_id - 1}`\n"
        f"- Target lost events: `{lost}`\n"
        f"- Target re-acquired events: `{reac}`\n"
        f"- Reports emitted: `{reports}`\n"
        f"- Robot max observed speed: `{robot.max_observed_speed:.2f} m/s`\n"
        f"- Requested actions: `{', '.join(graph.get('actions', []))}`"
    )


def persist_state(
    rgb: Optional[np.ndarray] = None,
    status: str = "",
    progress: float = 0.0,
    latest: str = "_Waiting for events..._",
    events_md: str = "_No events yet._",
    report_md: str = "_No report yet._",
    robot_md: str = "_Robot idle._",
    robot_log_md: str = "_No robot actions yet._",
    task_json: Optional[Dict[str, Any]] = None,
    metrics_md: str = "",
) -> None:
    if rgb is not None:
        st.session_state.last_frame = rgb
    st.session_state.last_status = status
    st.session_state.last_progress = progress
    st.session_state.latest_event_md = latest
    st.session_state.last_events_md = events_md
    st.session_state.last_report_md = report_md
    st.session_state.last_robot_md = robot_md
    st.session_state.last_robot_log_md = robot_log_md
    if task_json is not None:
        st.session_state.last_task_json = task_json
    if metrics_md:
        st.session_state.last_metrics_md = metrics_md


# -----------------------------
# Video encoding
# -----------------------------
def encode_frame_range(start: int, end: int) -> bytes:
    start = max(0, int(start))
    end = min(N_FRAMES, int(end))
    if end <= start:
        return b""

    # Preferred: H.264 MP4 via imageio-ffmpeg
    try:
        import imageio.v2 as imageio

        out = io.BytesIO()
        with imageio.get_writer(
            out,
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for i in range(start, end):
                frame = make_frame(i)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        data = out.getvalue()
        if data:
            return data
    except Exception:
        pass

    # Fallback: OpenCV MP4
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    for fourcc_code in ("avc1", "mp4v"):
        try:
            writer = cv2.VideoWriter(
                tmp.name,
                cv2.VideoWriter_fourcc(*fourcc_code),
                FPS,
                (WIDTH, HEIGHT),
            )
            if not writer.isOpened():
                continue

            for i in range(start, end):
                writer.write(make_frame(i))
            writer.release()

            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                with open(tmp.name, "rb") as f:
                    data = f.read()
                os.unlink(tmp.name)
                return data
        except Exception:
            continue

    try:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    except Exception:
        pass

    return b""


@st.cache_data(show_spinner=False)
def make_clip(start_t: float, end_t: float) -> bytes:
    start = int(max(0.0, float(start_t)) * FPS)
    end = int(min(float(SECONDS), float(end_t)) * FPS) + 1
    return encode_frame_range(start, end)


@st.cache_data(show_spinner=False)
def build_demo_assets() -> Tuple[bytes, List[Dict[str, Any]]]:
    video_bytes = encode_frame_range(0, N_FRAMES)

    archive_index: List[Dict[str, Any]] = []
    for i in range(N_FRAMES):
        frame = make_frame(i)
        dets = detect_objects(frame)
        zones = detect_zones(frame)

        objects = []
        for d in dets:
            objects.append({
                "label": d["label"],
                "color": d["color"],
                "shape": d["shape"],
                "center": [round(float(d["center"][0]), 2), round(float(d["center"][1]), 2)],
                "bbox": [int(v) for v in d["bbox"]],
                "conf": round(float(d["conf"]), 3),
            })

        active_zones = []
        for zn, zr in zones.items():
            if any(point_in_rect(o["center"], zr) for o in objects):
                active_zones.append(zn)

        caption = ", ".join(o["label"] for o in objects) if objects else "empty scene"
        if active_zones:
            caption += " | zones: " + ", ".join(active_zones)

        archive_index.append({
            "time": round(i / float(FPS), 2),
            "frame": i,
            "caption": caption,
            "objects": objects,
            "zones": {k: [int(v) for v in rect] for k, rect in zones.items()},
            "active_zones": active_zones,
        })

    return video_bytes, archive_index


def render_video_html(video_bytes: bytes, height: int = 270) -> bool:
    if not video_bytes:
        return False

    b64 = base64.b64encode(video_bytes).decode("ascii")
    html = f"""
    <video width="100%" height="{height}" controls autoplay muted loop playsinline>
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
      Your browser does not support HTML5 video.
    </video>
    """
    components.html(html, height=height + 24, scrolling=False)
    return True


# -----------------------------
# Video search index querying
# -----------------------------
def answer_search(graph: Dict[str, Any], archive_index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    attrs = graph["objects"][0]["attributes"]
    goal_region = graph.get("goal_region") or {}
    region = goal_region.get("type")

    has_enter = any(c.get("type") == "enter" for c in graph.get("conditions", []))
    if has_enter and not region:
        region = "marked_zone"

    raw_intervals: List[Dict[str, Any]] = []
    inside = False
    start_time = 0.0
    label = ""

    for entry in archive_index:
        matched = [o for o in entry["objects"] if attrs_match(attrs, o["color"], o["shape"])]
        presence = False

        if matched:
            if region is None:
                presence = True
            else:
                rect = entry["zones"].get(region)
                if rect:
                    presence = any(point_in_rect(o["center"], rect) for o in matched)

        if presence and not inside:
            inside = True
            start_time = float(entry["time"])
            label = matched[0]["label"]
        elif not presence and inside:
            inside = False
            raw_intervals.append({
                "start": start_time,
                "end": float(entry["time"]),
                "label": label,
            })

    if inside:
        raw_intervals.append({
            "start": start_time,
            "end": float(archive_index[-1]["time"]) if archive_index else float(SECONDS),
            "label": label,
        })

    merged: List[Dict[str, Any]] = []
    for inv in raw_intervals:
        if merged and inv["start"] - merged[-1]["end"] < 0.8:
            merged[-1]["end"] = inv["end"]
        else:
            merged.append(dict(inv))

    results: List[Dict[str, Any]] = []

    for inv in merged:
        if region:
            summary = f"{inv['label']} was inside {region} from {inv['start']:.1f}s to {inv['end']:.1f}s."
            rtype = "enter" if has_enter else "inside"
        else:
            summary = f"{inv['label']} was visible from {inv['start']:.1f}s to {inv['end']:.1f}s."
            rtype = "appearance"

        results.append({
            "type": rtype,
            "label": inv["label"],
            "region": region,
            "start": float(inv["start"]),
            "end": float(inv["end"]),
            "summary": summary,
        })

    if "count" in graph.get("actions", []):
        distinct_labels = sorted(set(inv["label"] for inv in merged))
        count = len(distinct_labels) if distinct_labels else 0
        results.insert(0, {
            "type": "count",
            "label": ", ".join(distinct_labels) if distinct_labels else "none",
            "region": region,
            "start": 0.0,
            "end": float(SECONDS),
            "summary": f"Counted {count} distinct matching object label(s): {', '.join(distinct_labels) if distinct_labels else 'none'}.",
        })

    return results


def process_search_query(query: str, archive_index: List[Dict[str, Any]]) -> None:
    graph, error = parse_instruction(query)
    if error:
        st.session_state.last_search_query = query
        st.session_state.last_search_error = error
        st.session_state.last_search_results = []
        return

    results = answer_search(graph, archive_index)
    st.session_state.last_search_query = query
    st.session_state.last_search_error = None
    st.session_state.last_search_results = results


def render_search_static() -> None:
    st.subheader("🔎 Video search index")

    query = st.session_state.get("last_search_query")
    err = st.session_state.get("last_search_error")
    results = st.session_state.get("last_search_results", [])

    if query:
        st.caption(f"Query: `{md_escape(query)}`")

    if err:
        st.error(err)
        return

    if not results:
        st.info("No matching timestamps. Try: 'When did a person in a yellow vest enter the dock?'")
        return

    for i, r in enumerate(results):
        st.markdown(f"**{str(r['type']).upper()}** — {md_escape(r['summary'])}")

        thumb_idx = int(np.clip(float(r.get("start", 0.0)) * FPS, 0, N_FRAMES - 1))
        thumb = make_frame(thumb_idx)
        st.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), use_container_width=True)

        if r["type"] != "count":
            clip_start = max(0.0, float(r["start"]) - 0.5)
            clip_end = min(float(SECONDS), max(clip_start + 1.0, min(float(r["end"]) + 0.5, clip_start + 4.0)))

            if st.button(f"🎬 Generate clip {i + 1}", key=f"gen_clip_{i}_{clip_start}_{clip_end}"):
                st.session_state.pending_clip = {
                    "start": float(clip_start),
                    "end": float(clip_end),
                    "name": f"search_clip_{i + 1}_{clip_start:.1f}-{clip_end:.1f}.mp4",
                }
                st.rerun()

    last_clip = st.session_state.get("last_clip")
    if last_clip and last_clip.get("bytes"):
        st.success("Generated search clip is ready.")
        st.video(last_clip["bytes"], format="video/mp4")
        st.download_button(
            "⬇️ Download generated clip",
            data=last_clip["bytes"],
            file_name=last_clip["name"],
            mime="video/mp4",
        )


# -----------------------------
# Simulation runner
# -----------------------------
def run_simulation(command: str, ui: Dict[str, Any], archive_index: List[Dict[str, Any]]) -> None:
    command_clean = command.replace("`", "'")
    graph, error = parse_instruction(command)

    task_payload = graph if graph else {"error": error}
    ui["task_box"].code(json.dumps(task_payload, indent=2), language="json")
    st.session_state.last_task_json = task_payload

    if error:
        status = f"❌ **Refused / invalid instruction:** {md_escape(error)}"
        ui["status_box"].markdown(status)
        persist_state(
            status=status,
            progress=st.session_state.get("last_progress", 0.0),
            latest=st.session_state.get("latest_event_md", "_Waiting for events..._"),
            events_md="_No events._",
            report_md="_No report._",
            robot_md="_Robot idle._",
            robot_log_md="_No robot actions._",
            task_json=task_payload,
        )
        return

    if "search" in graph["actions"] and "follow" not in graph["actions"]:
        process_search_query(command, archive_index)
        status = "✅ Search instruction routed to the video archive index. Open the **Video search** tab."
        ui["status_box"].markdown(status)
        persist_state(status=status, task_json=graph)
        st.rerun()
        return

    tracker = Tracker()
    robot = RobotAPI(
        max_speed=float(st.session_state.get("max_speed_mps", 1.0)),
        min_distance=0.8,
        follow_distance=float(st.session_state.get("follow_distance_m", 1.2)),
    )
    monitor = Monitor()

    target_id: Optional[int] = None
    target_lost = False
    triggered: set[str] = set()
    count_ids: set[int] = set()
    latencies: List[float] = []
    state = "running"
    video_dt = 1.0 / float(FPS)

    for i in range(N_FRAMES):
        if st.session_state.get("stop_simulation") or st.session_state.get("emergency_stop"):
            state = "stopped"
            monitor.add_event("stopped_by_user", "Operator stopped the simulation.", i, "warning")
            break

        t0 = time.perf_counter()

        frame = make_frame(i)
        dets = detect_objects(frame)
        zones = detect_zones(frame)
        active_tracks = tracker.update(dets, i)

        if "count" in graph["actions"]:
            attrs = graph["objects"][0]["attributes"]
            for tr in active_tracks:
                if attrs_match(attrs, tr.color, tr.shape):
                    count_ids.add(tr.id)

        if target_id is None and any(a in graph["actions"] for a in ("find", "follow", "report")):
            target_track, ambiguity = resolve_target(graph, active_tracks, zones, command)

            if ambiguity:
                status = f"⚠️ **Clarification needed:** {md_escape(ambiguity['question'])}"
                vis = draw_overlay(frame, dets, active_tracks, zones, None, robot, monitor,
                                   bool(st.session_state.get("show_debug_overlay", True)))
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

                ui["live_image"].image(rgb, use_container_width=True)
                ui["status_box"].markdown(status)
                ui["progress_box"].progress((i + 1) / float(N_FRAMES))
                ui["latest_event_box"].markdown(latest_event_md(monitor))
                ui["event_box"].markdown(events_to_md(monitor.events))
                ui["report_box"].markdown(report_to_md(monitor.events))
                ui["robot_box"].markdown(robot_to_md(robot, None, 0.0))
                ui["robot_log_box"].markdown(robot_log_to_md(robot))

                persist_state(
                    rgb=rgb,
                    status=status,
                    progress=(i + 1) / float(N_FRAMES),
                    latest=latest_event_md(monitor),
                    events_md=events_to_md(monitor.events),
                    report_md=report_to_md(monitor.events),
                    robot_md=robot_to_md(robot, None, 0.0),
                    robot_log_md=robot_log_to_md(robot),
                    task_json=graph,
                )

                st.session_state.clarification = ambiguity
                return

            if target_track:
                target_id = target_track.id
                monitor.add_event("target_acquired", f"Locked on {target_track.label} (ID {target_id}).", i)

        target = tracker.get(target_id)
        target_present = target is not None and target.active and target.missing == 0

        if target_id is not None:
            if target_present and target_lost:
                monitor.add_event("target_reacquired", f"Re-identified {target.label} (ID {target_id}).", i)
                target_lost = False
            elif not target_present and not target_lost:
                lost_label = target.label if target is not None else f"ID {target_id}"
                monitor.add_event("target_lost", f"Target {lost_label} lost.", i, "warning")
                target_lost = True

        speed = 0.0
        events_before = len(monitor.events)

        if target_present:
            speed = update_spatial_state(target, zones, video_dt)
            evaluate_conditions(graph, target, zones, i, triggered, monitor)

        if "follow" in graph["actions"]:
            control_robot(robot, target if target_present else None, active_tracks, graph, video_dt, monitor, i)

        if len(monitor.events) > events_before:
            for e in monitor.events[events_before:]:
                if e["type"] in ("report", "entered_zone", "target_lost", "target_reacquired", "count_report", "safety_stop"):
                    robot.speak(e["message"])

        show_debug = bool(st.session_state.get("show_debug_overlay", True))
        vis = draw_overlay(frame, dets, active_tracks, zones, target_id if target_present else None, robot, monitor, show_debug)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        target_label = target.label if target_present else ("lost" if target_id is not None else "searching")
        status = (
            f"**Task:** `{md_escape(command_clean)}`  \n"
            f"**State:** {md_escape(state)}  \n"
            f"**Frame:** {i + 1}/{N_FRAMES} ({i / float(FPS):.1f}s)  \n"
            f"**Target:** {md_escape(target_label)}  \n"
            f"**Robot pose:** x={robot.pose[0]:.2f} m, y={robot.pose[1]:.2f} m, θ={degrees(robot.pose[2]):.0f}°  \n"
            f"**Robot speed:** {robot.speed:.2f} m/s  \n"
            f"**Events:** {len(monitor.events)}"
        )

        ui["live_image"].image(rgb, use_container_width=True)
        ui["status_box"].markdown(status)
        ui["progress_box"].progress((i + 1) / float(N_FRAMES))
        ui["latest_event_box"].markdown(latest_event_md(monitor))
        ui["event_box"].markdown(events_to_md(monitor.events))
        ui["report_box"].markdown(report_to_md(monitor.events))
        ui["robot_box"].markdown(robot_to_md(robot, target if target_present else None, speed))
        ui["robot_log_box"].markdown(robot_log_to_md(robot))

        persist_state(
            rgb=rgb,
            status=status,
            progress=(i + 1) / float(N_FRAMES),
            latest=latest_event_md(monitor),
            events_md=events_to_md(monitor.events),
            report_md=report_to_md(monitor.events),
            robot_md=robot_to_md(robot, target if target_present else None, speed),
            robot_log_md=robot_log_to_md(robot),
            task_json=graph,
        )

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        playback_fps = float(st.session_state.get("playback_fps", FPS))
        sleep_time = max(0.0, 1.0 / max(1.0, playback_fps) - elapsed)
        time.sleep(sleep_time)

    if state == "running":
        if "count" in graph["actions"]:
            msg = f"Counted {len(count_ids)} matching object(s)."
            monitor.add_event("count_report", msg, N_FRAMES - 1, "success")
            robot.speak(msg)

        has_report = any(e["type"] == "report" for e in monitor.events)
        if "report" in graph["actions"] and not has_report:
            if any(e["type"] == "task_timeout" for e in monitor.events):
                monitor.add_event("task_timeout", "No qualifying event before timeout.", N_FRAMES - 1, "warning")
            else:
                monitor.add_event("task_complete", "Task window finished without a trigger.", N_FRAMES - 1)
        else:
            monitor.add_event("task_complete", "Task completed.", N_FRAMES - 1, "success")

    final_frame = make_frame(min(N_FRAMES - 1, max(0, len(latencies) - 1)))
    final_vis = draw_overlay(
        final_frame,
        detect_objects(final_frame),
        tracker.active_tracks(),
        detect_zones(final_frame),
        target_id,
        robot,
        monitor,
        bool(st.session_state.get("show_debug_overlay", True)),
    )
    final_rgb = cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)

    final_status = (
        f"**Task:** `{md_escape(command_clean)}`  \n"
        f"**State:** {md_escape(state)}  \n"
        f"**Frames processed:** {len(latencies)}/{N_FRAMES}  \n"
        f"**Reports:** {sum(1 for e in monitor.events if e['type'] == 'report')}  \n"
        f"**Open Video search / Events tabs for details.**"
    )

    metrics_md = metrics_to_md(latencies, monitor, tracker, robot, graph, state)

    ui["live_image"].image(final_rgb, use_container_width=True)
    ui["status_box"].markdown(final_status)
    ui["progress_box"].progress(1.0)
    ui["latest_event_box"].markdown(latest_event_md(monitor))
    ui["event_box"].markdown(events_to_md(monitor.events))
    ui["report_box"].markdown(report_to_md(monitor.events))
    ui["robot_box"].markdown(robot_to_md(robot, tracker.get(target_id), 0.0))
    ui["robot_log_box"].markdown(robot_log_to_md(robot))
    ui["metric_box"].markdown(metrics_md)

    persist_state(
        rgb=final_rgb,
        status=final_status,
        progress=1.0,
        latest=latest_event_md(monitor),
        events_md=events_to_md(monitor.events),
        report_md=report_to_md(monitor.events),
        robot_md=robot_to_md(robot, tracker.get(target_id), 0.0),
        robot_log_md=robot_log_to_md(robot),
        task_json=graph,
        metrics_md=metrics_md,
    )

    st.session_state.stop_simulation = False
    st.session_state.emergency_stop = False


def render_clarification(container: Any) -> None:
    cl = st.session_state.get("clarification")
    if not cl:
        return

    with container:
        st.warning(cl["question"])
        cols = st.columns(min(3, max(1, len(cl["options"]))))
        for idx, opt in enumerate(cl["options"]):
            with cols[idx % len(cols)]:
                if st.button(opt["label"], key=f"clarify_{idx}_{opt['label']}"):
                    st.session_state.pending_command = opt["new_instruction"]
                    st.session_state.clarification = None
                    st.session_state.stop_simulation = False
                    st.session_state.emergency_stop = False
                    st.rerun()


def restore_ui(ui: Dict[str, Any]) -> None:
    frame = st.session_state.get("last_frame")
    if frame is None:
        frame = cv2.cvtColor(make_frame(0), cv2.COLOR_BGR2RGB)

    ui["live_image"].image(frame, use_container_width=True)
    ui["status_box"].markdown(st.session_state.get("last_status", "_Ready. Auto-starting demo..._"))
    ui["progress_box"].progress(float(st.session_state.get("last_progress", 0.0)))
    ui["latest_event_box"].markdown(st.session_state.get("latest_event_md", "_Waiting for events..._"))

    task = st.session_state.get("last_task_json")
    if task is None:
        g, _ = parse_instruction(DEFAULT_COMMAND)
        task = g or {}
    ui["task_box"].code(json.dumps(task, indent=2), language="json")

    ui["event_box"].markdown(st.session_state.get("last_events_md", "_No events yet._"))
    ui["report_box"].markdown(st.session_state.get("last_report_md", "_No report yet._"))
    ui["robot_box"].markdown(st.session_state.get("last_robot_md", "_Robot idle._"))
    ui["robot_log_box"].markdown(st.session_state.get("last_robot_log_md", "_No robot actions yet._"))
    ui["metric_box"].markdown(st.session_state.get("last_metrics_md", "_Metrics available after demo run._"))


# -----------------------------
# Session initialization
# -----------------------------
def init_session() -> None:
    defaults = {
        "pending_command": None,
        "pending_search": None,
        "pending_clip": None,
        "last_clip": None,
        "stop_simulation": False,
        "emergency_stop": False,
        "auto_started": False,
        "clarification": None,
        "search_initialized": False,
        "last_search_results": [],
        "last_search_query": None,
        "last_search_error": None,
        "last_frame": None,
        "last_status": None,
        "last_progress": 0.0,
        "latest_event_md": "_Waiting for events..._",
        "last_events_md": "_No events yet._",
        "last_report_md": "_No report yet._",
        "last_robot_md": "_Robot idle._",
        "last_robot_log_md": "_No robot actions yet._",
        "last_metrics_md": "_Metrics available after demo run._",
        "last_task_json": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("🎛 Controls")

    sample = st.selectbox("Sample instruction", SAMPLE_COMMANDS, index=0)
    custom_instruction = st.text_area("Custom instruction", value=sample, height=88)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Run instruction", type="primary", use_container_width=True):
            st.session_state.pending_command = custom_instruction
            st.session_state.stop_simulation = False
            st.session_state.emergency_stop = False
            st.session_state.clarification = None
    with c2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.stop_simulation = True
            st.session_state.emergency_stop = True

    st.divider()

    st.subheader("Robot limits")
    st.slider("Playback FPS", min_value=4, max_value=24, value=FPS, key="playback_fps")
    st.slider("Follow distance (m)", min_value=0.6, max_value=3.0, value=1.2, step=0.1, key="follow_distance_m")
    st.slider("Max robot speed (m/s)", min_value=0.2, max_value=ABS_MAX_SPEED, value=1.0, step=0.1, key="max_speed_mps")
    st.checkbox("Show detector / tracker overlay", value=True, key="show_debug_overlay")

    st.divider()

    st.subheader("Video search")
    search_query = st.text_input("Archive query", value=DEFAULT_SEARCH_QUERY, key="search_query_input")
    if st.button("🔎 Search archive", use_container_width=True):
        st.session_state.pending_search = search_query

    st.divider()

    if st.button("♻️ Reset demo", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    st.caption("Local demo only. No API key. Robot actions are simulated and safety-limited.")


# -----------------------------
# Build demo assets
# -----------------------------
with st.spinner("Generating realistic demo video, archive index, and initial task graph..."):
    video_bytes, archive_index = build_demo_assets()

if not st.session_state.get("search_initialized", False):
    process_search_query(DEFAULT_SEARCH_QUERY, archive_index)
    st.session_state.search_initialized = True

if st.session_state.get("pending_search"):
    q = st.session_state.pending_search
    st.session_state.pending_search = None
    process_search_query(q, archive_index)

if st.session_state.get("pending_clip"):
    pc = st.session_state.pending_clip
    st.session_state.pending_clip = None
    clip = make_clip(float(pc["start"]), float(pc["end"]))
    st.session_state.last_clip = {"bytes": clip, "name": pc["name"]}

if not st.session_state.get("auto_started", False):
    st.session_state.auto_started = True
    if not st.session_state.get("pending_command"):
        st.session_state.pending_command = DEFAULT_COMMAND
        st.session_state.stop_simulation = False
        st.session_state.emergency_stop = False


# -----------------------------
# Main UI
# -----------------------------
st.title("🤖 Embodied Vision-Language Robotics Demo")
st.caption(
    "Natural-language instruction → validated task graph → visual grounding → tracking → spatial reasoning → "
    "event reporting → simulated robot actions. Fully local demo with generated video and no API key."
)

col_video, col_live = st.columns([1.05, 1.0])

with col_video:
    st.subheader("Generated demo video")
    if video_bytes:
        render_video_html(video_bytes, height=270)
        st.download_button(
            "⬇️ Download demo.mp4",
            data=video_bytes,
            file_name="embodied_vlm_demo.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.warning("MP4 encoder unavailable in this environment. The live simulation below still works.")
        st.image(cv2.cvtColor(make_frame(0), cv2.COLOR_BGR2RGB), use_container_width=True)

with col_live:
    st.subheader("Live simulated robot view")
    live_image = st.empty()
    status_box = st.empty()
    progress_box = st.empty()
    latest_event_box = st.empty()

ui = {
    "live_image": live_image,
    "status_box": status_box,
    "progress_box": progress_box,
    "latest_event_box": latest_event_box,
}

tab_task, tab_events, tab_search, tab_robot, tab_metrics = st.tabs(
    ["🧠 Task graph", "🔔 Events & report", "🔎 Video search", "🦿 Robot telemetry", "📊 Metrics"]
)

with tab_task:
    task_box = st.empty()
    ui["task_box"] = task_box

with tab_events:
    event_box = st.empty()
    report_box = st.empty()
    ui["event_box"] = event_box
    ui["report_box"] = report_box

with tab_search:
    render_search_static()

with tab_robot:
    robot_box = st.empty()
    robot_log_box = st.empty()
    ui["robot_box"] = robot_box
    ui["robot_log_box"] = robot_log_box

with tab_metrics:
    metric_box = st.empty()
    ui["metric_box"] = metric_box

clarification_container = st.container()

restore_ui(ui)

if st.session_state.get("pending_command"):
    cmd = st.session_state.pending_command
    st.session_state.pending_command = None
    run_simulation(cmd, ui, archive_index)

render_clarification(clarification_container)