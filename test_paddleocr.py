import os
import re
import glob
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def read_with_paddle(image_path: str):
    try:
        output = ocr.predict(image_path)

        texts = []

        for res in output:
            data = res.json if hasattr(res, "json") else res

            # some versions wrap output inside {"res": {...}}
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            if isinstance(data, dict):
                # general OCR pipeline style
                if "rec_texts" in data and data["rec_texts"]:
                    texts.extend([str(t) for t in data["rec_texts"] if t])

                # text recognition module style
                elif "rec_text" in data and data["rec_text"]:
                    texts.append(str(data["rec_text"]))

        raw_text = " ".join(texts).strip()
        clean_text = clean_plate_text(raw_text)

        return raw_text if raw_text else "ERR", clean_text if clean_text else "ERR"

    except Exception as e:
        print(f"[DEBUG] {image_path}: {type(e).__name__}: {e}")
        return "ERR", "ERR"

def main():
    image_paths = sorted(glob.glob("lpr_output/*.jpg"))

    if not image_paths:
        print("khong tim thay anh trong lpr_output")
        return

    # debug 1 anh truoc
    test_one = image_paths[0]
    print(f"[DEBUG] testing first image: {test_one}")
    try:
        sample = ocr.predict(test_one)
        for item in sample:
            data = item.json if hasattr(item, "json") else item
            print("[DEBUG] raw result =", data)
            break
    except Exception as e:
        print(f"[DEBUG] predict failed on first image: {type(e).__name__}: {e}")

    print("=" * 80)

    for path in image_paths:
        raw, clean = read_with_paddle(path)
        print(f"{os.path.basename(path)}")
        print(f"  raw   : {raw}")
        print(f"  clean : {clean}")
        print("-" * 50)

if __name__ == "__main__":
    main()