"""以 imgaug 擴增物件偵測資料，讀寫 YOLO TXT 標註。"""

import argparse
from pathlib import Path

from imgaug import augmenters as iaa

if __package__:
    from .augmentor import Augmentor
else:
    from augmentor import Augmentor


PROJECT_DIR = Path(__file__).resolve().parent.parent
TXT_DATASET = PROJECT_DIR / "Archive" / "bsh_aug_txt"
XML_DATASET = PROJECT_DIR / "Archive" / "bsh_aug"
FLIP_LIST = PROJECT_DIR / "flip" / "flip.txt"

# 每個操作獨立套用到原圖，可在此新增其他 imgaug augmenter。
TRANSFORMS = {
    "hflip": lambda: iaa.Fliplr(1.0),
    "vflip": lambda: iaa.Flipud(1.0),
    "rotate90": lambda: iaa.Rot90(1, keep_size=False),
    "rotate180": lambda: iaa.Rot90(2, keep_size=False),
    "rotate270": lambda: iaa.Rot90(3, keep_size=False),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, metavar="PATH",
        help="原圖資料夾（只讀取第一層），或每行一個影像路徑的清單。TXT 預設讀取 Archive/bsh_aug_txt，XML 預設讀取 Archive/bsh_aug。",
    )
    parser.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="輸出資料夾。內建範例的擴增圖預設寫入 Archive/bsh_aug_txt。自訂輸入或單純轉檔時，預設寫入原圖所在資料夾。",
    )
    parser.add_argument("--labels-dir", type=Path, help="指定來源標註資料夾。預設尋找與原圖搭配的同名標註。")
    parser.add_argument("--input-format", choices=("txt", "xml"), help="輸入標註格式。擴增預設為 txt，轉檔為 xml。")
    parser.add_argument("--classes-file", type=Path, help="類別清單，每行一個類別名稱，第一行的 ID 為 0。支援自動讀取標註旁的 classes.txt。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--transforms", nargs="+", choices=TRANSFORMS,
        help="按指定順序獨立擴增。預設 hflip vflip。輸出依序編號 3、4 等，重跑會更新同名擴增檔。",
    )
    mode.add_argument(
        "--convert-only", action="store_true",
        help="將 XML 轉為 YOLO TXT，預設將標註寫在原圖旁。指定其他輸出資料夾時會一併複製原圖。",
    )
    args = parser.parse_args(argv)
    if args.convert_only and args.input_format == "txt":
        parser.error("--convert-only 需要 XML 來源標註。")

    label_format = args.input_format or ("xml" if args.convert_only else "txt")
    source = args.input or (XML_DATASET if label_format == "xml" else TXT_DATASET)
    output_dir = args.output_dir
    if args.input is None and output_dir is None and not args.convert_only:
        output_dir = TXT_DATASET
    classes_file = args.classes_file
    if label_format == "xml" and classes_file is None and source.resolve() in (XML_DATASET, FLIP_LIST):
        if not ((args.labels_dir or XML_DATASET) / "classes.txt").is_file():
            classes_file = TXT_DATASET / "classes.txt"

    try:
        dataset = Augmentor(
            input_file=source,
            output_dir=output_dir,
            label_format=label_format,
            output_format="txt",
            labels_dir=args.labels_dir,
            classes_file=classes_file,
        )
        if args.convert_only:
            count = dataset.convert_to_txt()
            print(f"轉檔完成，共 {count} 組原圖與 YOLO TXT 標註。")
        else:
            dataset.readImages()
            for name in args.transforms or ("hflip", "vflip"):
                dataset.addAugmentor(TRANSFORMS[name]())
            count = dataset.augment()
            print(f"擴增完成，共 {count} 組擴增影像與 YOLO TXT 標註。")
    except (ValueError, OSError) as error:
        parser.exit(1, f"{parser.prog} 執行失敗。{error}\n")


if __name__ == "__main__":
    main()
