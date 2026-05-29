from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import json
import time
from collections import Counter
from typing import Optional, List
import threading

try:
    import kociemba
    KOCIEMBA_AVAILABLE = True
except ImportError:
    KOCIEMBA_AVAILABLE = False

app = FastAPI(title="Rubik's Cube Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Color detection logic (from your original code) ─────────────────────────

COLORS = {
    'White':  ([0,   0,   200], [180, 30,  255]),
    'Yellow': ([20,  80,  180], [35,  255, 255]),
    'Red':    ([0,   120, 120], [10,  255, 255]),
    'Red2':   ([170, 120, 120], [180, 255, 255]),
    'Green':  ([40,  50,  50],  [80,  255, 255]),
    'Blue':   ([100, 60,  60],  [130, 255, 255]),
    'Orange': ([5,   100, 180], [18,  255, 255]),
    'Orange2':([170, 100, 180], [180, 255, 255]),
}

FACE_META = {
    'U': {'name': 'TOP',    'center': 'White'},
    'D': {'name': 'BOTTOM', 'center': 'Red'},
    'F': {'name': 'FRONT',  'center': 'Yellow'},
    'B': {'name': 'BACK',   'center': 'Green'},
    'L': {'name': 'LEFT',   'center': 'Orange'},
    'R': {'name': 'RIGHT',  'center': 'Blue'},
}

# Global state
state = {
    'faces': {k: {'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False}
              for k in ['U', 'D', 'F', 'B', 'L', 'R']},
    'current_face': 'U',
    'face_order': ['U', 'D', 'F', 'B', 'L', 'R'],
    'face_index': 0,
}
state_lock = threading.Lock()

cap = None
cap_lock = threading.Lock()
camera_active = True   # tracks whether camera streaming is on

SQUARE_SIZE = 400


# ─── Request models ───────────────────────────────────────────────────────────

class ManualFaceInput(BaseModel):
    face_id: str                          # U / D / F / B / L / R
    colors: List[List[str]]               # 3×3 grid of color names


def get_camera():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
        return cap


def get_color_name(h, s, v):
    if s < 40 and v > 200:
        return 'White'
    if 80 < s and 180 < v:
        if 5 <= h <= 18 or h >= 170:
            return 'Orange'
    for name, (lo, hi) in COLORS.items():
        if '2' in name:
            continue
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return name
        if name == 'Red' and 170 <= h <= 180 and s >= 120 and v >= 120:
            return 'Red'
        if name == 'Orange' and 170 <= h <= 180 and s >= 100 and v >= 180:
            return 'Orange'
    return 'Unknown'


def detect_all_cells(frame, square_size=SQUARE_SIZE):
    h, w = frame.shape[:2]
    x1 = w // 2 - square_size // 2
    y1 = h // 2 - square_size // 2
    cell_w = square_size // 3
    cell_h = square_size // 3
    colors = [['?' for _ in range(3)] for _ in range(3)]

    for row in range(3):
        for col in range(3):
            pad = 8
            rx1 = x1 + col * cell_w + pad
            ry1 = y1 + row * cell_h + pad
            rx2 = x1 + (col + 1) * cell_w - pad
            ry2 = y1 + (row + 1) * cell_h - pad
            cell = frame[ry1:ry2, rx1:rx2]
            if cell.size == 0:
                continue
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            med = np.median(hsv.reshape(-1, 3), axis=0)
            colors[row][col] = get_color_name(*med)

    return colors


def build_kociemba_string(faces):
    COLOR_MAP = {
        'White': 'U', 'Blue': 'R', 'Yellow': 'F',
        'Red': 'D', 'Orange': 'L', 'Green': 'B',
    }
    parts = []
    for face in ['U', 'R', 'F', 'D', 'L', 'B']:
        fd = faces[face]
        if fd['captured']:
            for row in range(3):
                for col in range(3):
                    parts.append(COLOR_MAP.get(fd['colors'][row][col], 'U'))
        else:
            parts.extend(['U'] * 9)
    return ''.join(parts)


def is_solved(faces):
    """Return True if every face is a single uniform color (cube is already solved)."""
    for face_data in faces.values():
        if not face_data['captured']:
            return False
        colors = face_data['colors']
        flat = [c for row in colors for c in row]
        if len(set(flat)) != 1:
            return False
    return True


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "kociemba": KOCIEMBA_AVAILABLE}


@app.post("/camera/stop")
def stop_camera():
    """Release the camera so the user can enter colors manually."""
    global cap, camera_active
    with cap_lock:
        camera_active = False
        if cap and cap.isOpened():
            cap.release()
            cap = None
    return {"success": True, "camera_active": False}


@app.post("/camera/start")
def start_camera():
    """Re-open the camera for scanning."""
    global camera_active
    camera_active = True
    get_camera()          # re-opens cap
    return {"success": True, "camera_active": True}


@app.get("/camera/status")
def camera_status():
    return {"camera_active": camera_active}


@app.get("/camera/frame")
def get_frame():
    """Return current camera frame as base64 JPEG with overlay."""
    if not camera_active:
        raise HTTPException(status_code=409, detail="Camera is stopped")
    camera = get_camera()
    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    ret, frame = camera.read()
    if not ret:
        raise HTTPException(status_code=503, detail="Cannot read frame")

    # Draw grid overlay
    h, w = frame.shape[:2]
    sq = SQUARE_SIZE
    x1 = w // 2 - sq // 2
    y1 = h // 2 - sq // 2
    x2 = x1 + sq
    y2 = y1 + sq

    cell_w = sq // 3
    cell_h = sq // 3

    with state_lock:
        current = state['current_face']
        colors_grid = detect_all_cells(frame)

    COLOR_BGR = {
        'White': (255, 255, 255), 'Yellow': (0, 255, 255),
        'Red': (0, 0, 255), 'Green': (0, 200, 0),
        'Blue': (255, 0, 0), 'Orange': (0, 165, 255),
        'Unknown': (120, 120, 120),
    }

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for i in range(1, 3):
        cv2.line(frame, (x1 + i * cell_w, y1), (x1 + i * cell_w, y2), (0, 255, 0), 2)
        cv2.line(frame, (x1, y1 + i * cell_h), (x2, y1 + i * cell_h), (0, 255, 0), 2)

    for row in range(3):
        for col in range(3):
            cx = x1 + col * cell_w + cell_w // 2
            cy = y1 + row * cell_h + cell_h // 2
            color = colors_grid[row][col]
            bgr = COLOR_BGR.get(color, (120, 120, 120))
            cv2.rectangle(frame, (cx - 22, cy - 18), (cx + 22, cy + 18), bgr, -1)
            cv2.rectangle(frame, (cx - 22, cy - 18), (cx + 22, cy + 18), (0, 0, 0), 2)
            txt = color[:3].upper() if color != 'Unknown' else '?'
            cv2.putText(frame, txt, (cx - 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    face_meta = FACE_META[current]
    detected_center = colors_grid[1][1]
    center_ok = detected_center == face_meta['center']
    cv2.putText(frame, f"Scan: {face_meta['name']} (center={face_meta['center']})", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Detected center: {detected_center}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if center_ok else (0, 80, 255), 2)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    return {
        "image": b64,
        "current_face": current,
        "face_name": face_meta['name'],
        "expected_center": face_meta['center'],
        "detected_center": detected_center,
        "center_match": center_ok,
        "detected_colors": colors_grid,
    }


@app.post("/capture")
def capture_face():
    """Capture the current face using live camera."""
    if not camera_active:
        raise HTTPException(status_code=409, detail="Camera is stopped — use /manual-input instead")
    camera = get_camera()
    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    ret, frame = camera.read()
    if not ret:
        raise HTTPException(status_code=503, detail="Cannot read frame")

    with state_lock:
        current = state['current_face']
        face_meta = FACE_META[current]
        colors = detect_all_cells(frame)
        detected_center = colors[1][1]

        if detected_center != face_meta['center']:
            return {
                "success": False,
                "error": f"Wrong face! Expected center={face_meta['center']}, got {detected_center}",
                "detected_center": detected_center,
                "expected_center": face_meta['center'],
            }

        state['faces'][current]['colors'] = colors
        state['faces'][current]['captured'] = True

        captured = sum(1 for f in state['faces'].values() if f['captured'])
        all_done = captured == 6

        if not all_done and state['face_index'] < 5:
            state['face_index'] += 1
            state['current_face'] = state['face_order'][state['face_index']]

    return {
        "success": True,
        "face": current,
        "colors": colors,
        "captured_count": captured,
        "all_done": all_done,
        "next_face": state['current_face'] if not all_done else None,
    }


@app.post("/retake/{face_id}")
def retake_face(face_id: str):
    if face_id not in state['faces']:
        raise HTTPException(status_code=400, detail="Invalid face")
    with state_lock:
        state['faces'][face_id]['colors'] = [['?' for _ in range(3)] for _ in range(3)]
        state['faces'][face_id]['captured'] = False
        idx = state['face_order'].index(face_id)
        state['face_index'] = idx
        state['current_face'] = face_id
    return {"success": True, "current_face": face_id}


@app.post("/reset")
def reset_all():
    with state_lock:
        for fid in state['faces']:
            state['faces'][fid] = {'colors': [['?' for _ in range(3)] for _ in range(3)], 'captured': False}
        state['current_face'] = 'U'
        state['face_index'] = 0
    return {"success": True}


@app.get("/faces")
def get_faces():
    with state_lock:
        captured = sum(1 for f in state['faces'].values() if f['captured'])
        return {
            "faces": state['faces'],
            "current_face": state['current_face'],
            "captured_count": captured,
            "face_order": state['face_order'],
        }


@app.post("/manual-input")
def manual_input(data: ManualFaceInput):
    """Accept a manually entered 3×3 color grid for one face."""
    fid = data.face_id.upper()
    if fid not in state['faces']:
        raise HTTPException(status_code=400, detail=f"Invalid face id: {fid}")

    if len(data.colors) != 3 or any(len(row) != 3 for row in data.colors):
        raise HTTPException(status_code=400, detail="colors must be a 3×3 grid")

    VALID = {'White', 'Yellow', 'Red', 'Green', 'Blue', 'Orange'}
    for row in data.colors:
        for c in row:
            if c not in VALID:
                raise HTTPException(status_code=400, detail=f"Invalid color: {c}")

    expected_center = FACE_META[fid]['center']
    detected_center = data.colors[1][1]
    if detected_center != expected_center:
        raise HTTPException(
            status_code=400,
            detail=f"Center mismatch: {fid} face center must be {expected_center}, got {detected_center}"
        )

    with state_lock:
        state['faces'][fid]['colors'] = [list(row) for row in data.colors]
        state['faces'][fid]['captured'] = True
        captured = sum(1 for f in state['faces'].values() if f['captured'])
        all_done = captured == 6

        # Auto-advance current_face pointer if this face was next in order
        if fid == state['current_face'] and not all_done and state['face_index'] < 5:
            state['face_index'] += 1
            state['current_face'] = state['face_order'][state['face_index']]

    return {
        "success": True,
        "face": fid,
        "colors": data.colors,
        "captured_count": captured,
        "all_done": all_done,
    }


@app.get("/is-solved")
def check_is_solved():
    """Check whether the scanned cube is already in the solved state."""
    with state_lock:
        captured = sum(1 for f in state['faces'].values() if f['captured'])
        if captured < 6:
            return {"solved": False, "reason": f"Only {captured}/6 faces captured", "captured": captured}
        solved = is_solved(state['faces'])
        return {
            "solved": solved,
            "reason": "Cube is already solved!" if solved else "Cube needs solving",
            "captured": captured,
        }


@app.post("/solve")
def solve():
    with state_lock:
        captured = sum(1 for f in state['faces'].values() if f['captured'])
        if captured < 6:
            raise HTTPException(status_code=400, detail=f"Only {captured}/6 faces captured")

        if not KOCIEMBA_AVAILABLE:
            raise HTTPException(status_code=503, detail="kociemba not installed. Run: pip install kociemba")

        # Short-circuit: cube already solved
        if is_solved(state['faces']):
            return {"success": True, "moves": [], "move_count": 0, "already_solved": True, "cube_string": ""}

        cube_str = build_kociemba_string(state['faces'])

        if len(cube_str) != 54:
            raise HTTPException(status_code=400, detail="Invalid cube string")

        # Validate color counts
        counts = Counter(cube_str)
        for letter in 'URFDLB':
            if counts.get(letter, 0) != 9:
                raise HTTPException(status_code=400, detail=f"Color imbalance: {letter}={counts.get(letter,0)}")

        try:
            solution = kociemba.solve(cube_str)
            moves = solution.strip().split()
            return {"success": True, "moves": moves, "move_count": len(moves), "cube_string": cube_str}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.on_event("shutdown")
def shutdown():
    global cap
    if cap and cap.isOpened():
        cap.release()