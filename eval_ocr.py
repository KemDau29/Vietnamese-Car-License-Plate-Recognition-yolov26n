import csv
import re
import time
from pathlib import Path

import easyocr
from paddleocr import PaddleOCR

DATASET_DIR = Path("bien_so_xe_may_dataset")
IMAGE_DIR = DATASET_DIR / "anh"
LABEL_DIR = DATASET_DIR / "label"
RESULT_CSV = Path("ocr_eval_results.csv")


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 1.0 if pred else 0.0
    return levenshtein(pred, gt) / len(gt)


def load_dataset_pairs():
    rows = []

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"khong tim thay anh trong {IMAGE_DIR}")

    for img_path in image_paths:
        txt_path = LABEL_DIR / f"{img_path.stem}.txt"
        if not txt_path.exists():
            print(f"[skip] missing label: {img_path.name}")
            continue

        gt_raw = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        gt = normalize_plate(gt_raw)

        if not gt:
            print(f"[skip] empty label: {txt_path.name}")
            continue

        rows.append({
            "filename": img_path.name,
            "image_path": str(img_path),
            "gt_raw": gt_raw,
            "gt": gt,
        })

    return rows


def init_easyocr():
    return easyocr.Reader(["en"], gpu=False)


def init_paddleocr():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )


def read_easyocr(reader, image_path: str):
    try:
        texts = reader.readtext(image_path, detail=0)
        raw = " ".join(texts).strip()
        clean = normalize_plate(raw)
        return raw if raw else "ERR", clean if clean else "ERR"
    except Exception as e:
        return f"ERR: {e}", "ERR"


def read_paddleocr(ocr, image_path: str):
    try:
        output = ocr.predict(image_path)
        texts = []

        for res in output:
            data = res.json if hasattr(res, "json") else res

            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            if isinstance(data, dict):
                if "rec_texts" in data and data["rec_texts"]:
                    texts.extend([str(x) for x in data["rec_texts"] if x])
                elif "rec_text" in data and data["rec_text"]:
                    texts.append(str(data["rec_text"]))

        raw = " ".join(texts).strip()
        clean = normalize_plate(raw)
        return raw if raw else "ERR", clean if clean else "ERR"

    except Exception as e:
        return f"ERR: {e}", "ERR"


def summarize(rows, prefix: str):
    n = len(rows)
    exact = sum(r[f"{prefix}_exact"] for r in rows) / n
    avg_cer = sum(r[f"{prefix}_cer"] for r in rows) / n
    avg_time = sum(r[f"{prefix}_time_ms"] for r in rows) / n

    print(f"{prefix} exact_match_acc : {exact:.4f}")
    print(f"{prefix} avg_cer         : {avg_cer:.4f}")
    print(f"{prefix} char_acc        : {1 - avg_cer:.4f}")
    print(f"{prefix} avg_time_ms     : {avg_time:.2f}")


def main():
    rows = load_dataset_pairs()
    if not rows:
        print("khong co cap anh-label hop le")
        return

    print(f"so mau benchmark: {len(rows)}")

    print("init easyocr...")
    easy_reader = init_easyocr()

    print("init paddleocr...")
    paddle_ocr = init_paddleocr()

    first_img = rows[0]["image_path"]
    print(f"warm-up voi {rows[0]['filename']}")
    _ = read_easyocr(easy_reader, first_img)
    _ = read_paddleocr(paddle_ocr, first_img)

    result_rows = []

    for row in rows:
        image_path = row["image_path"]
        gt = row["gt"]

        t0 = time.perf_counter()
        easy_raw, easy_clean = read_easyocr(easy_reader, image_path)
        easy_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        paddle_raw, paddle_clean = read_paddleocr(paddle_ocr, image_path)
        paddle_time = (time.perf_counter() - t1) * 1000

        result_rows.append({
            "filename": row["filename"],
            "gt_raw": row["gt_raw"],
            "gt": gt,
            "easy_raw": easy_raw,
            "easy_clean": easy_clean,
            "easy_exact": int(easy_clean == gt),
            "easy_cer": cer(easy_clean, gt),
            "easy_time_ms": round(easy_time, 2),
            "paddle_raw": paddle_raw,
            "paddle_clean": paddle_clean,
            "paddle_exact": int(paddle_clean == gt),
            "paddle_cer": cer(paddle_clean, gt),
            "paddle_time_ms": round(paddle_time, 2),
        })

    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    print("\n=== summary ===")
    summarize(result_rows, "easy")
    print()
    summarize(result_rows, "paddle")

    print(f"\nda luu chi tiet vao: {RESULT_CSV}")


if __name__ == "__main__":
    main()
