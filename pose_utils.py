"""
Core logic for the yoga pose checker:
  1. Body keypoint "skeletonization"  (MediaPipe Pose, 33 landmarks)
  2. Joint-angle calculation          (angle at a joint from 3 landmarks)
  3. Pose scoring + plain-English feedback
  4. Drawing the skeleton + angles on the image
"""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

AUTO = "Auto-detect"
VIS_MIN = 0.5  # landmarks less visible than this are ignored

# --------------------------------------------------------------------------
# 1. Joint definitions: joint -> (point A, vertex B, point C); angle is at B
# --------------------------------------------------------------------------
def _lm(side: str, part: str):
    return getattr(PL, f"{side.upper()}_{part}")


JOINT_DEFS: dict[str, tuple] = {}
for _side in ("left", "right"):
    JOINT_DEFS[f"{_side}_elbow"] = (_lm(_side, "SHOULDER"), _lm(_side, "ELBOW"), _lm(_side, "WRIST"))
    JOINT_DEFS[f"{_side}_shoulder"] = (_lm(_side, "ELBOW"), _lm(_side, "SHOULDER"), _lm(_side, "HIP"))
    JOINT_DEFS[f"{_side}_hip"] = (_lm(_side, "SHOULDER"), _lm(_side, "HIP"), _lm(_side, "KNEE"))
    JOINT_DEFS[f"{_side}_knee"] = (_lm(_side, "HIP"), _lm(_side, "KNEE"), _lm(_side, "ANKLE"))

# --------------------------------------------------------------------------
# 2. Pose library: joint -> (target angle in degrees, tolerance in degrees)
#    These are reasonable starting values, NOT medical/teaching standards.
#    "view" says which camera angle gives reliable 2D angles; "aliases"
#    are short names accepted by live.py --pose and resolve_pose().
#    "mirror": True means the pose can be done on either side, so both
#    left/right assignments are tried and the better one is used.
# --------------------------------------------------------------------------
POSES: dict[str, dict] = {
    # ---------------- standing poses: shoot from the FRONT ----------------
    "Mountain Pose": {
        "mirror": False, "view": "front", "aliases": ["mountain", "tadasana"],
        "tips": "Stand tall, feet together, legs straight, arms relaxed by your sides.",
        "joints": {
            "left_knee": (180, 10), "right_knee": (180, 10),
            "left_hip": (180, 10), "right_hip": (180, 10),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (20, 20), "right_shoulder": (20, 20),
        },
    },
    "Tree Pose (arms overhead)": {
        "mirror": True, "view": "front", "aliases": ["tree", "vrikshasana", "vrksasana"],
        "tips": "Stand on one leg, other foot on inner thigh/calf, arms straight overhead.",
        "joints": {
            "left_knee": (180, 15), "right_knee": (45, 25),
            "left_hip": (180, 15),
            "left_elbow": (180, 25), "right_elbow": (180, 25),
            "left_shoulder": (165, 30), "right_shoulder": (165, 30),
        },
    },
    "Chair Pose": {
        "mirror": False, "view": "front", "aliases": ["chair", "utkatasana"],
        "tips": "Sit back as if on a chair, knees ~110 deg, arms straight overhead.",
        "joints": {
            "left_knee": (110, 20), "right_knee": (110, 20),
            "left_hip": (100, 20), "right_hip": (100, 20),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (160, 25), "right_shoulder": (160, 25),
        },
    },
    "Goddess Pose": {
        "mirror": False, "view": "front", "aliases": ["goddess", "utkata konasana"],
        "tips": "Wide squat, toes out, thighs near parallel; elbows bent 90 deg, palms up (cactus arms).",
        "joints": {
            "left_knee": (100, 20), "right_knee": (100, 20),
            "left_hip": (110, 25), "right_hip": (110, 25),
            "left_elbow": (90, 20), "right_elbow": (90, 20),
            "left_shoulder": (90, 25), "right_shoulder": (90, 25),
        },
    },
    "Warrior I": {
        "mirror": True, "view": "front",
        "aliases": ["warrior1", "warrior 1", "warrior i", "virabhadrasana 1"],
        "tips": "Front knee bent ~90 deg, back leg straight, hips square, arms straight overhead.",
        "joints": {
            "left_knee": (90, 15), "right_knee": (170, 15),
            "left_hip": (100, 25), "right_hip": (160, 20),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (165, 25), "right_shoulder": (165, 25),
        },
    },
    "Warrior II": {
        "mirror": True, "view": "front",
        "aliases": ["warrior2", "warrior 2", "warrior ii", "warrior", "virabhadrasana 2"],
        "tips": "Wide stance, front knee bent ~90 deg, back leg straight, arms parallel to floor.",
        "joints": {
            "left_knee": (90, 15), "right_knee": (180, 12),
            "left_hip": (95, 25), "right_hip": (165, 20),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (90, 15), "right_shoulder": (90, 15),
        },
    },
    "Triangle Pose": {
        "mirror": False, "view": "front", "aliases": ["triangle", "trikonasana"],
        "tips": "Wide straight legs, torso bent sideways, arms in one vertical line.",
        "joints": {
            "left_knee": (180, 12), "right_knee": (180, 12),
            "left_hip": (100, 30), "right_hip": (100, 30),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (90, 25), "right_shoulder": (90, 25),
        },
    },
    "Low Lunge": {
        "mirror": True, "view": "side", "aliases": ["lunge", "low lunge", "anjaneyasana"],
        "tips": "Front knee ~90 deg over the ankle, back knee on the floor, arms straight overhead.",
        "joints": {
            "left_knee": (90, 15), "right_knee": (120, 25),
            "left_hip": (100, 25), "right_hip": (160, 20),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (165, 25), "right_shoulder": (165, 25),
        },
    },
    # ---------------- floor poses: shoot from the SIDE ----------------
    "Downward Dog": {
        "mirror": False, "view": "side", "aliases": ["downdog", "down dog", "downward", "adho mukha"],
        "tips": "Inverted V: arms and legs straight, hips high.",
        "joints": {
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_knee": (180, 15), "right_knee": (180, 15),
            "left_shoulder": (170, 20), "right_shoulder": (170, 20),
            "left_hip": (75, 20), "right_hip": (75, 20),
        },
    },
    "Upward Dog": {
        "mirror": False, "view": "side", "aliases": ["updog", "up dog", "upward", "urdhva mukha"],
        "tips": "Arms straight, chest lifted, thighs and knees off the floor, legs straight.",
        "joints": {
            "left_elbow": (180, 12), "right_elbow": (180, 12),
            "left_shoulder": (35, 25), "right_shoulder": (35, 25),
            "left_hip": (145, 25), "right_hip": (145, 25),
            "left_knee": (175, 15), "right_knee": (175, 15),
        },
    },
    "Cobra Pose": {
        "mirror": False, "view": "side", "aliases": ["cobra", "bhujangasana"],
        "tips": "Lie on your front, pelvis down, elbows softly bent, chest lifted.",
        "joints": {
            "left_elbow": (145, 25), "right_elbow": (145, 25),
            "left_shoulder": (30, 20), "right_shoulder": (30, 20),
            "left_hip": (145, 20), "right_hip": (145, 20),
            "left_knee": (180, 15), "right_knee": (180, 15),
        },
    },
    "Plank": {
        "mirror": False, "view": "side", "aliases": ["plank", "phalakasana", "kumbhakasana"],
        "tips": "Straight line head to heels, arms straight under shoulders.",
        "joints": {
            "left_elbow": (180, 12), "right_elbow": (180, 12),
            "left_shoulder": (90, 20), "right_shoulder": (90, 20),
            "left_hip": (180, 12), "right_hip": (180, 12),
            "left_knee": (180, 12), "right_knee": (180, 12),
        },
    },
    "Bridge Pose": {
        "mirror": False, "view": "side", "aliases": ["bridge", "setu bandha"],
        "tips": "Lie on your back, feet near hips, lift hips so shoulders-hips-knees form a line; arms on the floor.",
        "joints": {
            "left_knee": (65, 20), "right_knee": (65, 20),
            "left_hip": (170, 15), "right_hip": (170, 15),
            "left_elbow": (180, 20), "right_elbow": (180, 20),
            "left_shoulder": (30, 25), "right_shoulder": (30, 25),
        },
    },
    "Boat Pose": {
        "mirror": False, "view": "side", "aliases": ["boat", "navasana"],
        "tips": "Balance on sit bones, legs straight and lifted, arms straight forward parallel to floor.",
        "joints": {
            "left_knee": (170, 20), "right_knee": (170, 20),
            "left_hip": (65, 20), "right_hip": (65, 20),
            "left_elbow": (180, 15), "right_elbow": (180, 15),
            "left_shoulder": (70, 25), "right_shoulder": (70, 25),
        },
    },
    "Child's Pose": {
        "mirror": False, "view": "side", "aliases": ["child", "childs", "balasana"],
        "tips": "Kneel, hips to heels, forehead down, arms straight and reaching forward.",
        "joints": {
            "left_knee": (35, 20), "right_knee": (35, 20),
            "left_hip": (40, 25), "right_hip": (40, 25),
            "left_elbow": (180, 20), "right_elbow": (180, 20),
            "left_shoulder": (170, 25), "right_shoulder": (170, 25),
        },
    },
}


def _mirror_name(name: str) -> str:
    side, part = name.split("_", 1)
    return f"{'right' if side == 'left' else 'left'}_{part}"


# --------------------------------------------------------------------------
# Pose estimator wrapper (context manager)
# --------------------------------------------------------------------------
class PoseEstimator:
    def __init__(self, static: bool = True):
        self._pose = mp_pose.Pose(
            static_image_mode=static,
            model_complexity=2 if static else 1,
            smooth_landmarks=not static,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image_rgb: np.ndarray):
        return self._pose.process(image_rgb)

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# 3. Angle calculation
# --------------------------------------------------------------------------
def angle_between(a, b, c) -> float | None:
    """Angle in degrees at point b formed by a-b-c (2D)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def compute_angles(landmarks, w: int, h: int) -> dict:
    """Return {joint: {angle, visibility, point}} using pixel coordinates."""
    out = {}
    for name, (ia, ib, ic) in JOINT_DEFS.items():
        la, lb, lc = landmarks[ia], landmarks[ib], landmarks[ic]
        pa, pb, pc = (la.x * w, la.y * h), (lb.x * w, lb.y * h), (lc.x * w, lc.y * h)
        ang = angle_between(pa, pb, pc)
        if ang is None:
            continue
        out[name] = {
            "angle": ang,
            "visibility": min(la.visibility, lb.visibility, lc.visibility),
            "point": (int(pb[0]), int(pb[1])),
        }
    return out


# --------------------------------------------------------------------------
# 4. Scoring + feedback
# --------------------------------------------------------------------------
def score_pose(angles: dict, spec: dict) -> dict:
    variants = [spec["joints"]]
    if spec.get("mirror"):
        variants.append({_mirror_name(k): v for k, v in spec["joints"].items()})

    best = None
    for joints in variants:
        results = []
        for name, (target, tol) in joints.items():
            a = angles.get(name)
            if a is None or a["visibility"] < VIS_MIN:
                results.append({"joint": name, "angle": None, "target": target,
                                "tol": tol, "dev": None, "score": None, "ok": None})
                continue
            dev = abs(a["angle"] - target)
            score = 100.0 if dev <= tol else max(0.0, 100.0 * (1 - (dev - tol) / 60.0))
            results.append({"joint": name, "angle": a["angle"], "target": target,
                            "tol": tol, "dev": dev, "score": score, "ok": dev <= tol})
        seen = [r for r in results if r["score"] is not None]
        overall = float(np.mean([r["score"] for r in seen])) if seen else 0.0
        cand = {"results": results, "overall": overall, "n_visible": len(seen),
                "n_total": len(results)}
        if best is None or cand["overall"] > best["overall"]:
            best = cand
    return best


def _hint(joint: str, cur: float, target: float) -> str:
    side, kind = joint.split("_", 1)
    more = cur < target
    info = f"(now {cur:.0f}\u00b0, aim for ~{target:.0f}\u00b0)"
    if kind in ("knee", "elbow"):
        verb = "Straighten" if more else "Bend"
        return f"{verb} your {side} {kind} {'more ' if not more else ''}{info}."
    if kind == "shoulder":
        verb = "Raise your {s} arm higher" if more else "Lower your {s} arm"
        return verb.format(s=side) + f" {info}."
    if kind == "hip":
        if more:
            return f"Open your {side} hip \u2013 lengthen the line from shoulder to knee {info}."
        return f"Fold/sink deeper at your {side} hip {info}."
    return f"Adjust your {side} {kind} {info}."


def build_feedback(evaluation: dict) -> list[str]:
    bad = [r for r in evaluation["results"] if r["ok"] is False]
    bad.sort(key=lambda r: r["dev"] - r["tol"], reverse=True)
    return [_hint(r["joint"], r["angle"], r["target"]) for r in bad[:5]]


def analyze(image_rgb: np.ndarray, pose_name: str, estimator: PoseEstimator) -> dict | None:
    """Run the full pipeline. Returns None if no person is found."""
    h, w = image_rgb.shape[:2]
    res = estimator.process(image_rgb)
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks.landmark
    angles = compute_angles(lms, w, h)

    ranking = []
    if pose_name == AUTO:
        for n, spec in POSES.items():
            ev = score_pose(angles, spec)
            if ev["n_visible"] >= 3:
                ranking.append((n, ev))
        ranking.sort(key=lambda x: x[1]["overall"], reverse=True)
        if not ranking:
            return {"landmarks": lms, "angles": angles, "pose": None,
                    "evaluation": None, "ranking": [], "confident": False, "feedback": []}
        chosen, evaluation = ranking[0]
        # Auto-detect is only trustworthy when most joints were measured and the
        # winner is clearly ahead of the runner-up.
        margin = evaluation["overall"] - (ranking[1][1]["overall"] if len(ranking) > 1 else 0.0)
        confident = (evaluation["n_visible"] >= evaluation["n_total"] / 2
                     and margin >= 5.0 and evaluation["overall"] >= 60.0)
    else:
        chosen, evaluation = pose_name, score_pose(angles, POSES[pose_name])
        confident = True

    return {"landmarks": lms, "angles": angles, "pose": chosen,
            "evaluation": evaluation, "ranking": ranking, "confident": confident,
            "feedback": build_feedback(evaluation)}


# --------------------------------------------------------------------------
# 5. Drawing (works on an RGB image; colours are RGB)
# --------------------------------------------------------------------------
def draw_overlay(image_rgb: np.ndarray, analysis: dict) -> np.ndarray:
    img = image_rgb.copy()
    h, w = img.shape[:2]
    s = max(w, h) / 1000.0
    lms = analysis["landmarks"]

    # skeleton bones
    for i, j in mp_pose.POSE_CONNECTIONS:
        if lms[i].visibility < 0.3 or lms[j].visibility < 0.3:
            continue
        p1 = (int(lms[i].x * w), int(lms[i].y * h))
        p2 = (int(lms[j].x * w), int(lms[j].y * h))
        cv2.line(img, p1, p2, (240, 240, 240), max(2, int(3 * s)), cv2.LINE_AA)
    # keypoints
    for lm in lms:
        if lm.visibility < 0.3:
            continue
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), max(3, int(5 * s)),
                   (60, 120, 255), -1, cv2.LINE_AA)

    # angle labels: green = within tolerance, red = needs correction
    status = {}
    if analysis["evaluation"]:
        status = {r["joint"]: r["ok"] for r in analysis["evaluation"]["results"]}
    for name, a in analysis["angles"].items():
        if name not in status or status[name] is None:
            continue
        color = (40, 200, 80) if status[name] else (235, 50, 50)
        x, y = a["point"]
        cv2.circle(img, (x, y), max(6, int(9 * s)), color, -1, cv2.LINE_AA)
        cv2.putText(img, f"{a['angle']:.0f}", (x + int(10 * s), y - int(10 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), max(4, int(5 * s)), cv2.LINE_AA)
        cv2.putText(img, f"{a['angle']:.0f}", (x + int(10 * s), y - int(10 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, color, max(2, int(2 * s)), cv2.LINE_AA)
    return img


def resolve_pose(query: str) -> str:
    """Match a short name/alias like 'tree', 'warrior1' or 'cobra' to a pose in POSES."""
    q = (query or "").strip().lower()
    if not q or q in ("auto", AUTO.lower()):
        return AUTO
    for n, spec in POSES.items():                       # exact name or alias
        if q == n.lower() or q in spec.get("aliases", []):
            return n
    hits = [n for n in POSES if q in n.lower()]         # unique substring of the full name
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"'{query}' is ambiguous: {', '.join(hits)}")
    raise ValueError(f"Unknown pose '{query}'. Options: {', '.join(POSES)}")
