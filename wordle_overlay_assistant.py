import math
import os
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pyautogui
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from PyQt6 import QtCore, QtGui, QtWidgets

# ---------------- CONFIG -----------------
ROWS, COLS = 6, 5
UPDATE_INTERVAL_MS = 900
SCREENSHOT_INTERVAL = 4
MAX_SAVED_SCREENSHOTS = 120

DEFAULT_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(DEFAULT_TESSERACT_WIN):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_WIN

SAMPLES_DIR = Path(os.getcwd()) / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

WORDLIST_FILE = Path(os.getcwd()) / "wordlist_wordle.txt"
WINDOWS_VALID_WORDS_CSV = Path(
    r"C:\Users\Liam\OneDrive\Desktop\Advacned cool looking, spotify playlist thing\valid-words.csv"
)

GREEN_CANDIDATES = [(83, 141, 78), (106, 170, 100)]
YELLOW_CANDIDATES = [(181, 159, 59), (201, 180, 88)]
GREY_CANDIDATES = [(58, 58, 60), (120, 124, 126)]
WHITE_CANDIDATES = [(255, 255, 255), (248, 248, 248), (230, 230, 230)]
UNKNOWN_COLOR = "unknown"
COLOR_TOLERANCE = 72

COLOR_TO_PATTERN = {"green": "g", "yellow": "y", "grey": "b"}

MIN_SUBMITTED_COLORS = 5

FALLBACK_WORDS = [
    "slate", "crane", "trace", "stare", "adieu", "audio", "raise", "irate", "arise", "later", "alter",
    "alert", "tears", "rates", "stale", "least", "learn", "snare", "react", "roate", "soare", "lares",
    "plane", "store", "stone", "shine", "spare", "spine", "shale", "chore", "glare", "flame", "grace",
    "pride", "chair", "sugar", "bring", "about", "other", "which", "there", "their", "could", "would",
    "sound", "house", "water", "light", "world", "heart", "cover", "point", "round", "beach", "sleep",
    "write", "quick", "brown", "foxes", "jumps", "vodka", "fuzzy", "glyph", "nymph", "brick", "cigar",
    "rebut", "sissy", "humph", "awake", "blush", "focal", "evade", "naval", "serve", "heath", "dwarf",
    "model", "karma", "grade", "quiet", "bench", "abate", "feign", "major", "death", "fresh", "crust",
]


@dataclass
class TileDetection:
    x: int
    y: int
    left: int
    top: int
    right: int
    bottom: int
    letter: str
    color_name: str
    row: int
    col: int


# ---------------- Detection utilities -----------------
def dist_rgb(a: Sequence[int], b: Sequence[int]) -> float:
    return math.sqrt(sum((int(a[i]) - int(b[i])) ** 2 for i in range(3)))


def nearest_palette_name(rgb: Tuple[int, int, int]) -> str:
    groups = {
        "green": GREEN_CANDIDATES,
        "yellow": YELLOW_CANDIDATES,
        "grey": GREY_CANDIDATES,
        "white": WHITE_CANDIDATES,
    }
    best = (float("inf"), UNKNOWN_COLOR)
    for name, candidates in groups.items():
        for c in candidates:
            d = dist_rgb(rgb, c)
            if d < best[0]:
                best = (d, name)

    if best[0] <= COLOR_TOLERANCE:
        return best[1]

    brightness = sum(rgb) / 3
    spread = max(rgb) - min(rgb)
    if brightness > 210:
        return "white"
    if spread < 14 and brightness < 150:
        return "grey"
    return UNKNOWN_COLOR


def average_rgb(img: Image.Image) -> Tuple[int, int, int]:
    px = list(img.getdata())
    if not px:
        return (0, 0, 0)
    return (
        sum(p[0] for p in px) // len(px),
        sum(p[1] for p in px) // len(px),
        sum(p[2] for p in px) // len(px),
    )


def preprocess_for_ocr(img: Image.Image) -> List[Image.Image]:
    gray = img.convert("L")
    boosted = ImageEnhance.Contrast(gray).enhance(2.8)
    variants = [
        boosted,
        ImageOps.invert(boosted),
        boosted.point(lambda p: 255 if p > 120 else 0),
        boosted.point(lambda p: 255 if p > 145 else 0),
        ImageOps.invert(boosted.point(lambda p: 255 if p > 120 else 0)),
    ]
    return [v.resize((v.width * 3, v.height * 3), Image.Resampling.BICUBIC) for v in variants]


def build_letter_templates(size: int = 40) -> Dict[str, Image.Image]:
    templates: Dict[str, Image.Image] = {}
    font = ImageFont.load_default()
    for ch in string.ascii_uppercase:
        img = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((size // 3, size // 4), ch, fill=255, font=font)
        templates[ch] = img
    return templates


def tensor_similarity_letter(tile_img: Image.Image, templates: Dict[str, Image.Image]) -> str:
    proc = ImageEnhance.Contrast(tile_img.convert("L")).enhance(3.0).resize((40, 40), Image.Resampling.BICUBIC)
    proc = proc.point(lambda p: 255 if p > 120 else 0)
    a = list(proc.getdata())

    best_score = -1.0
    best_letter = "E"
    for ch, t in templates.items():
        b = list(t.getdata())
        dot = sum(float(a[i]) * float(b[i]) for i in range(len(a)))
        norm_a = math.sqrt(sum(float(v) * float(v) for v in a))
        norm_b = math.sqrt(sum(float(v) * float(v) for v in b))
        score = dot / (norm_a * norm_b + 1e-9)
        if score > best_score:
            best_score = score
            best_letter = ch
    return best_letter


def ocr_single_letter(tile_img: Image.Image, templates: Dict[str, Image.Image]) -> str:
    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for candidate in preprocess_for_ocr(tile_img):
        text = pytesseract.image_to_string(candidate, config=config).strip().upper()
        letters = "".join(ch for ch in text if "A" <= ch <= "Z")
        if letters:
            return letters[0]
    return tensor_similarity_letter(tile_img, templates)


def cleanup_old_screenshots(samples_dir: Path, keep: int) -> None:
    files = sorted(samples_dir.glob("wordle_overlay_*.png"), key=lambda p: p.stat().st_mtime)
    for old_file in files[: max(0, len(files) - keep)]:
        try:
            old_file.unlink()
            print(f"Deleted old screenshot: {old_file}")
        except OSError as exc:
            print(f"Could not delete {old_file}: {exc}")


def _extract_words_from_text(text: str) -> List[str]:
    out: List[str] = []
    normalized = text.replace("\r", "\n")
    for raw in normalized.split("\n"):
        for piece in raw.split(","):
            w = piece.strip().strip('"').lower()
            if len(w) == 5 and w.isalpha():
                out.append(w)
    return out


def load_words() -> List[str]:
    words: List[str] = []
    if WINDOWS_VALID_WORDS_CSV.exists():
        words.extend(_extract_words_from_text(WINDOWS_VALID_WORDS_CSV.read_text(encoding="utf-8", errors="ignore")))
    if not words and WORDLIST_FILE.exists():
        words.extend(_extract_words_from_text(WORDLIST_FILE.read_text(encoding="utf-8", errors="ignore")))
    if not words:
        words = [w.lower() for w in FALLBACK_WORDS if len(w) == 5 and w.isalpha()]
    return sorted(set(words))


# ---------------- Solver -----------------
def pattern_for_guess(guess: str, answer: str) -> str:
    result = ["b"] * 5
    answer_chars = list(answer)
    for i, ch in enumerate(guess):
        if ch == answer_chars[i]:
            result[i] = "g"
            answer_chars[i] = "*"
    for i, ch in enumerate(guess):
        if result[i] == "g":
            continue
        if ch in answer_chars:
            result[i] = "y"
            answer_chars[answer_chars.index(ch)] = "*"
    return "".join(result)


class WordleSolver:
    def __init__(self, words: List[str]):
        self.all_words = sorted(set(words))
        self.candidates = list(self.all_words)

    def recompute_from_feedbacks(self, feedback_rows: List[Tuple[str, str]]) -> None:
        cands = list(self.all_words)
        for guess, pattern in feedback_rows:
            cands = [w for w in cands if pattern_for_guess(guess, w) == pattern]
        self.candidates = cands

    def guess_scores(self) -> List[Tuple[str, float]]:
        if not self.candidates:
            return [("raise", 1.0)]

        search_space = self.all_words if len(self.candidates) > 1 else self.candidates
        total = float(len(self.candidates))
        scored: List[Tuple[str, float]] = []

        for guess in search_space:
            buckets: Dict[str, int] = {}
            for ans in self.candidates:
                p = pattern_for_guess(guess, ans)
                buckets[p] = buckets.get(p, 0) + 1

            entropy = 0.0
            for count in buckets.values():
                prob = count / total
                entropy -= prob * math.log2(max(prob, 1e-12))

            in_candidate_bonus = 0.3 if guess in self.candidates else 0.0
            score = entropy + in_candidate_bonus
            scored.append((guess, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def best_guess(self) -> str:
        return self.guess_scores()[0][0]

    def top_candidates_with_probability(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Estimate likelihood among current candidates using positional+letter frequency model."""
        if not self.candidates:
            return [("raise", 100.0)]

        pos_freq = [dict() for _ in range(5)]
        letter_freq = {}
        for w in self.candidates:
            for i, ch in enumerate(w):
                pos_freq[i][ch] = pos_freq[i].get(ch, 0) + 1
            for ch in set(w):
                letter_freq[ch] = letter_freq.get(ch, 0) + 1

        raw_scores: List[Tuple[str, float]] = []
        for w in self.candidates:
            s = 0.0
            for i, ch in enumerate(w):
                s += pos_freq[i].get(ch, 0)
            s += 0.4 * sum(letter_freq.get(ch, 0) for ch in set(w))
            raw_scores.append((w, s))

        # softmax for percentage
        max_s = max(s for _, s in raw_scores)
        exp_scores = [(w, math.exp(s - max_s)) for w, s in raw_scores]
        total = sum(v for _, v in exp_scores) or 1.0
        probs = [(w, 100.0 * v / total) for w, v in exp_scores]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_n]


# ---------------- GUI -----------------
class SolverDashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wordle Solver Dashboard")
        self.setMinimumSize(700, 520)

        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel("Waiting for calibration/detections...")
        self.summary.setStyleSheet("font-size:14px;font-weight:bold;color:#00d9ff;")
        layout.addWidget(self.summary)

        grid_and_list = QtWidgets.QHBoxLayout()
        layout.addLayout(grid_and_list)

        self.table = QtWidgets.QTableWidget(ROWS, COLS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        grid_and_list.addWidget(self.table, 2)

        right = QtWidgets.QVBoxLayout()
        grid_and_list.addLayout(right, 1)
        right.addWidget(QtWidgets.QLabel("Top likely words"))
        self.prob_list = QtWidgets.QListWidget()
        right.addWidget(self.prob_list)

        for r in range(ROWS):
            for c in range(COLS):
                it = QtWidgets.QTableWidgetItem(" ")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                it.setBackground(QtGui.QBrush(QtGui.QColor(40, 40, 40)))
                it.setForeground(QtGui.QBrush(QtGui.QColor("white")))
                self.table.setItem(r, c, it)

        self.rows_log = QtWidgets.QPlainTextEdit()
        self.rows_log.setReadOnly(True)
        self.rows_log.setPlaceholderText("Row log will appear here...")
        layout.addWidget(self.rows_log)

    @staticmethod
    def _cell_color(name: str) -> QtGui.QColor:
        if name == "green":
            return QtGui.QColor(83, 141, 78)
        if name == "yellow":
            return QtGui.QColor(181, 159, 59)
        if name == "grey":
            return QtGui.QColor(120, 124, 126)
        return QtGui.QColor(50, 50, 50)

    def update_state(
        self,
        row_logs: List[Tuple[str, List[str]]],
        suggestion: str,
        candidate_count: int,
        top_probs: List[Tuple[str, float]],
    ) -> None:
        self.summary.setText(f"Next suggestion: {suggestion.upper()} | Candidate count: {candidate_count}")
        lines = []

        for r, (letters, colors) in enumerate(row_logs):
            lines.append(f"Row {r+1} = {letters} = {colors}")
            for c in range(COLS):
                item = self.table.item(r, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.table.setItem(r, c, item)
                ch = letters[c] if c < len(letters) else " "
                color = colors[c] if c < len(colors) else UNKNOWN_COLOR
                item.setText(ch)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QtGui.QBrush(self._cell_color(color)))
                item.setForeground(QtGui.QBrush(QtGui.QColor("white")))

        self.rows_log.setPlainText("\n".join(lines) if lines else "No rows detected yet.")

        self.prob_list.clear()
        for word, pct in top_probs:
            self.prob_list.addItem(f"{word.upper()}  —  {pct:.1f}%")


class CountdownOverlay(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, seconds: int = 3, text: str = "Move mouse"):
        super().__init__()
        self.current = seconds
        self.text = text
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(QtWidgets.QApplication.primaryScreen().geometry())

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)
        self.show()
        self.raise_()

    def update_countdown(self) -> None:
        self.current -= 1
        self.update()
        if self.current < 0:
            self.timer.stop()
            self.close()
            self.finished.emit()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtGui.QFont("Arial", 92, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QPen(QtGui.QColor("yellow")))
        painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, f"{self.text}\n{max(0, self.current)}")


class CalibrationOverlay(QtWidgets.QWidget):
    position_recorded = QtCore.pyqtSignal(tuple)

    def __init__(self, corner_name: str):
        super().__init__()
        self.corner_name = corner_name
        self.dot_pos = None
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(QtWidgets.QApplication.primaryScreen().geometry())
        self.show()
        self.raise_()

        self.countdown = CountdownOverlay(3, f"Move mouse to {corner_name}")
        self.countdown.finished.connect(self.capture_position)

    def capture_position(self) -> None:
        pos = pyautogui.position()
        self.dot_pos = pos
        print(f"{self.corner_name} locked at ({pos.x}, {pos.y})")
        self.position_recorded.emit((pos.x, pos.y))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self.dot_pos:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))
        painter.setPen(QtGui.QPen(QtGui.QColor("red")))
        painter.drawEllipse(self.dot_pos.x - 8, self.dot_pos.y - 8, 16, 16)
        painter.setFont(QtGui.QFont("Arial", 30))
        painter.setPen(QtGui.QPen(QtGui.QColor("yellow")))
        painter.drawText(40, 60, f"{self.corner_name} locked")


class DetectionWorker(QtCore.QObject):
    result_ready = QtCore.pyqtSignal(object)
    stopped = QtCore.pyqtSignal()

    def __init__(self, corners: List[Tuple[int, int]]):
        super().__init__()
        self.top_left, self.top_right, self.bottom_left, self.bottom_right = corners
        self.col_spacing = (self.top_right[0] - self.top_left[0]) / (COLS - 1)
        self.row_spacing = (self.bottom_left[1] - self.top_left[1]) / (ROWS - 1)
        self.tile_size = max(24, int(min(abs(self.col_spacing), abs(self.row_spacing)) * 0.82))
        self.half_tile = self.tile_size // 2

        self.letter_templates = build_letter_templates()
        self.solver = WordleSolver(load_words())
        self.running = True
        self.tick = 0

    def stop(self) -> None:
        self.running = False

    def _tile_center(self, row: int, col: int) -> Tuple[int, int]:
        return int(self.top_left[0] + col * self.col_spacing), int(self.top_left[1] + row * self.row_spacing)

    def _tile_box(self, x: int, y: int) -> Tuple[int, int, int, int]:
        return x - self.half_tile, y - self.half_tile, x + self.half_tile, y + self.half_tile

    @staticmethod
    def _row_pattern(colors: List[str]) -> str:
        return "".join(COLOR_TO_PATTERN[c] for c in colors if c in COLOR_TO_PATTERN)

    @staticmethod
    def _played_row(letters: str, colors: List[str]) -> bool:
        if not (len(letters) == 5 and letters.isalpha() and len(colors) == 5):
            return False
        # submitted row should have all 5 feedback colors in the Wordle set
        return sum(1 for c in colors if c in COLOR_TO_PATTERN) >= MIN_SUBMITTED_COLORS

    def _feedback_rows(self, row_logs: List[Tuple[str, List[str]]]) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        for letters, colors in row_logs:
            low = letters.lower()
            if self._played_row(low, colors):
                rows.append((low, self._row_pattern(colors)))
        seen = set()
        uniq = []
        for r in rows:
            if r not in seen:
                uniq.append(r)
                seen.add(r)
        return uniq

    def _recompute_solver_safe(self, feedback_rows: List[Tuple[str, str]]) -> None:
        prev = list(self.solver.candidates)
        self.solver.recompute_from_feedbacks(feedback_rows)
        # Guard against OCR noise causing impossible state.
        if feedback_rows and not self.solver.candidates:
            print("Solver warning: OCR feedback led to 0 candidates, keeping previous candidate set.")
            self.solver.candidates = prev if prev else list(self.solver.all_words)

    def run(self) -> None:
        print(f"Loaded {len(self.solver.all_words)} words. First suggestion: {self.solver.best_guess().upper()}")
        while self.running:
            start = time.time()
            screenshot = pyautogui.screenshot()

            detections: List[TileDetection] = []
            row_logs: List[Tuple[str, List[str]]] = []

            for row in range(ROWS):
                letters: List[str] = []
                colors: List[str] = []
                for col in range(COLS):
                    x, y = self._tile_center(row, col)
                    l, t, r, b = self._tile_box(x, y)
                    tile = screenshot.crop((l, t, r, b))

                    color_name = nearest_palette_name(average_rgb(tile))
                    letter = ocr_single_letter(tile, self.letter_templates)

                    letters.append(letter)
                    colors.append(color_name)
                    detections.append(TileDetection(x, y, l, t, r, b, letter, color_name, row + 1, col + 1))

                row_logs.append(("".join(letters), colors))

            for idx, (letters, colors) in enumerate(row_logs, start=1):
                print(f"Row {idx}: {letters} -> {colors}")
            if not row_logs:
                print("Row detection loop running, but no rows parsed.")

            feedback_rows = self._feedback_rows(row_logs)
            self._recompute_solver_safe(feedback_rows)
            suggestion = self.solver.best_guess()
            top_probs = self.solver.top_candidates_with_probability(top_n=5)
            print(
                f"Solver: rows={feedback_rows} | candidates={len(self.solver.candidates)} | "
                f"next={suggestion.upper()} | likely={top_probs}"
            )

            self.tick += 1
            if self.tick % SCREENSHOT_INTERVAL == 0:
                self._save_annotated_screenshot(screenshot, row_logs, suggestion)

            self.result_ready.emit(
                {
                    "detections": detections,
                    "row_logs": row_logs,
                    "suggestion": suggestion,
                    "candidate_count": len(self.solver.candidates),
                    "top_probs": top_probs,
                }
            )

            elapsed_ms = (time.time() - start) * 1000.0
            sleep_ms = max(50.0, UPDATE_INTERVAL_MS - elapsed_ms)
            QtCore.QThread.msleep(int(sleep_ms))

        self.stopped.emit()

    def _save_annotated_screenshot(self, screenshot: Image.Image, row_logs: List[Tuple[str, List[str]]], suggestion: str) -> None:
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle((20, 20, 620, 60), fill=(0, 0, 0))
        draw.text((28, 28), f"Solver suggestion: {suggestion.upper()} ({len(self.solver.candidates)} cands)", fill=(0, 255, 255))

        top = self.solver.top_candidates_with_probability(top_n=3)
        draw.rectangle((20, 65, 620, 130), fill=(0, 0, 0))
        draw.text(
            (28, 72),
            "Likely: " + ", ".join(f"{w.upper()} {p:.1f}%" for w, p in top),
            fill=(255, 255, 0),
        )

        for row_idx, (letters, colors) in enumerate(row_logs):
            y = int(self.top_left[1] + row_idx * self.row_spacing)
            x = int(self.top_left[0] - 210)
            label = f"R{row_idx+1}: {letters} | {','.join(colors)}"
            draw.rectangle((x - 8, y - 54, x + 420, y - 20), fill=(0, 0, 0))
            draw.text((x, y - 50), label, fill=(255, 255, 0))

        for d in self.result_snapshot_detections(row_logs):
            draw.rectangle((d.left, d.top, d.right, d.bottom), outline=(255, 50, 50), width=3)
            draw.text((d.left + 2, d.top + 2), d.letter, fill=(0, 255, 255))

        path = SAMPLES_DIR / f"wordle_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img.save(path)
        print(f"Saved annotated screenshot: {path}")
        cleanup_old_screenshots(SAMPLES_DIR, MAX_SAVED_SCREENSHOTS)

    def result_snapshot_detections(self, row_logs: List[Tuple[str, List[str]]]) -> List[TileDetection]:
        out: List[TileDetection] = []
        for row in range(ROWS):
            for col in range(COLS):
                x, y = self._tile_center(row, col)
                l, t, r, b = self._tile_box(x, y)
                letter = row_logs[row][0][col] if col < len(row_logs[row][0]) else "?"
                color = row_logs[row][1][col] if col < len(row_logs[row][1]) else UNKNOWN_COLOR
                out.append(TileDetection(x, y, l, t, r, b, letter, color, row + 1, col + 1))
        return out


class WordleOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.detections: List[TileDetection] = []
        self.suggestion = "raise"
        self.candidate_count = 0

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background:rgba(0,0,0,0);")
        self.setGeometry(QtWidgets.QApplication.primaryScreen().geometry())
        self.show()
        self.raise_()

    def update_state(self, detections: List[TileDetection], suggestion: str, candidate_count: int) -> None:
        self.detections = detections
        self.suggestion = suggestion
        self.candidate_count = candidate_count
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255)))
        painter.drawText(30, 40, f"Try: {self.suggestion.upper()} | Cands: {self.candidate_count}")

        if not self.detections:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80), 2))
            painter.drawText(30, 70, "Waiting for detection stream...")
        for d in self.detections:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(d.left, d.top, d.right - d.left, d.bottom - d.top)
            painter.setFont(QtGui.QFont("Consolas", 10))
            painter.setPen(QtGui.QPen(QtGui.QColor("white")))
            painter.drawText(d.left + 2, d.top - 4, f"R{d.row}C{d.col}")
            painter.drawText(d.left + 2, d.bottom + 12, d.letter)


def start_worker(corners: List[Tuple[int, int]], overlay: WordleOverlay, dashboard: SolverDashboard) -> Tuple[QtCore.QThread, DetectionWorker]:
    thread = QtCore.QThread()
    worker = DetectionWorker(corners)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)

    def on_result(payload: object) -> None:
        data = payload
        overlay.update_state(data["detections"], data["suggestion"], data["candidate_count"])
        dashboard.update_state(data["row_logs"], data["suggestion"], data["candidate_count"], data["top_probs"])

    worker.result_ready.connect(on_result)
    worker.stopped.connect(thread.quit)

    thread.start()
    return thread, worker


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    corners_names = ["Top-left", "Top-right", "Bottom-left", "Bottom-right"]
    corner_positions: List[Tuple[int, int]] = []

    overlay = WordleOverlay()
    dashboard = SolverDashboard()
    dashboard.show()
    overlay.hide()

    state: Dict[str, object] = {"thread": None, "worker": None, "calibrators": []}

    def next_corner(index: int = 0) -> None:
        if index >= len(corners_names):
            print(f"Calibration complete: {corner_positions}")
            print("Starting background detection worker...")
            overlay.show()
            thread, worker = start_worker(corner_positions, overlay, dashboard)
            state["thread"] = thread
            state["worker"] = worker
            return

        calibrator = CalibrationOverlay(corners_names[index])
        state["calibrators"].append(calibrator)

        def pos_received(pos: Tuple[int, int]) -> None:
            corner_positions.append(pos)
            calibrator.close()
            state["calibrators"] = [c for c in state["calibrators"] if c is not calibrator]
            next_corner(index + 1)

        calibrator.position_recorded.connect(pos_received)

    next_corner()

    def cleanup() -> None:
        worker = state.get("worker")
        thread = state.get("thread")
        if worker is not None:
            worker.stop()
        if thread is not None:
            thread.quit()
            thread.wait(2000)

    app.aboutToQuit.connect(cleanup)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
