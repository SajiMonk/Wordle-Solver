# Wordle Overlay Assistant

`wordle_overlay_assistant.py` is a PyQt6 transparent overlay + advanced Wordle solver assistant.

## What it does now
- 4-corner calibration with 3-second countdown per corner.
- Click-through transparent overlay (you still type directly into Wordle).
- Uniform **square** tile boxes.
- Real-time per-tile detection:
  - OCR letters with Tesseract + preprocessing + tensor-style fallback inference
  - tile color classification (`green/yellow/grey/white/unknown`)
- Responsive architecture:
  - heavy screen capture + OCR + solving runs in a background worker thread
  - GUI stays responsive while updates stream in.
- Advanced solving loop:
  - rebuilds candidate set from all detected played rows (`green/yellow/grey` patterns)
  - entropy-style next-guess scoring to aggressively narrow candidates
  - computes top likely candidate words with percentage estimates.
- Advanced dashboard GUI:
  - row-by-row detected words
  - per-letter colored cells (green/yellow/grey)
  - next suggestion + candidate count (`cands` = remaining candidate words)
  - top likely words with percentages
  - row log panel
- Annotated screenshot export to `samples/` and automatic cleanup of old files.

## Solver word list
- If this Windows CSV exists, it is used first: `C:\Users\Liam\OneDrive\Desktop\Advacned cool looking, spotify playlist thing\valid-words.csv`.
- Otherwise if `wordlist_wordle.txt` exists in the project root, it is used.
- Otherwise the script uses a built-in fallback list of common 5-letter words.

## Run
```bash
python3 wordle_overlay_assistant.py
```

## Requirements
```bash
pip install PyQt6 pyautogui pillow pytesseract
```

Install Tesseract OCR and ensure it is in your `PATH`.
On Windows, the script auto-detects:
`C:\Program Files\Tesseract-OCR\tesseract.exe`.
