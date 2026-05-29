# Vietnamese Car License Plate Recognition (YOLOv26n)

## Overview

This project recognizes Vietnamese vehicle license plates by combining:
- object detection using `YOLO` with `ultralytics` and the `best.pt` model
- OCR-based text extraction via the `OCR.space` API
- a live UI with ROI control, barrier status, and recognition logs

The main system entrypoint is `main.py`.

## Features

- License plate detection inside a configurable ROI using `best.pt`
- OCR text recognition via `ocr_service.py`
- image preprocessing with `dip_algorithm_pro` in `image_processing.py`
- logging output to `lpr_output/lpr_log.csv`
- optional Arduino barrier control over COM serial
- offline OCR benchmarking via `eval_ocr.py` using `EasyOCR` and `PaddleOCR`

## Project Structure

- `main.py` - main runtime: captures camera stream, detects plates, controls barrier logic
- `ocr_service.py` - OCR.space API wrapper for plate text extraction
- `config.py` - global configuration, save paths, Arduino port, UI settings
- `image_processing.py` - image preprocessing using CLAHE
- `eval_ocr.py` - OCR evaluation script for offline accuracy benchmarking
- `best.pt` - YOLO model for license plate detection
- `yolo26n.pt` - YOLO model for vehicle detection

## Requirements

- Python 3.10+ (recommended 3.11+)
- `pip install -r requirements.txt`
- a camera stream supporting MJPEG/HTTP
- optional Arduino attached via COM port for barrier control
- valid OCR.space API key in a `.env` file

## Installation

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file in the repository root:

```env
OCR_API_KEY=your_ocr_space_api_key_here
```

3. If using Arduino, verify or update `ARDUINO_PORT` in `config.py`.

4. If using a phone camera stream, update `ip_address` in `main.py` to the correct device IP.

## Running the System

```bash
python main.py
```

When running, the system will:
- load video from `http://<ip_address>:8080/video`
- display a window with ROI controls
- detect license plates in the ROI
- call OCR.space to recognize plate text
- save recognition logs to `lpr_output/lpr_log.csv`

## Configuration

- ROI and confidence threshold (`CONF %`) can be adjusted in the on-screen config window
- `main.py` uses:
  - `best.pt` for license plate detection
  - `yolo26n.pt` for vehicle detection
- `config.py` defines:
  - `WINDOW_NAME`
  - `SAVE_DIR`
  - `CSV_FILE`
  - `ARDUINO_PORT`
  - `BAUD_RATE`

## OCR Evaluation

To benchmark OCR performance offline, run:

```bash
python eval_ocr.py
```

This script compares `EasyOCR` and `PaddleOCR`, then writes results to `ocr_eval_results.csv`.

## Notes

- `OCR.space` is an external service and requires a valid `OCR_API_KEY`.
- Camera stream settings and Arduino configuration depend on your actual hardware.
- Both `best.pt` and `yolo26n.pt` models are included in the repository.

## Contact

This project was developed by the HCMUTE team for a Vietnamese license plate recognition research project.

