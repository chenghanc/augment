# augment

使用 imgaug 同步變換影像與物件框，支援 YOLO TXT 擴增與 Pascal VOC XML 轉檔。

## 建立 Python 3.12 環境

在已安裝 [uv](https://docs.astral.sh/uv/) 的 macOS 或 Linux 上，從專案根目錄執行。若已有 Python 3.12 的 `venv`，可從 `source venv/bin/activate` 開始。

```bash
uv python install 3.12
uv venv --python 3.12 --seed venv
source venv/bin/activate
python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "scipy<1.18" \
  "imgaug==0.4.0"
python -m pip check
```

## 從 YOLO TXT 產生擴增資料

將原圖及同名 `.txt` 放入 `Archive/bsh_aug_txt/`。每個物件一行，格式為 `class_id center_x center_y width height`。類別從 0 編號，座標與框大小以影像寬高正規化到 0 至 1。沒有物件的圖片搭配空白標註檔。

```bash
python flip/demo_API.py
```

每張圖會產生水平與垂直翻轉各一組。例如 `frame.jpg` 的輸出位於 `Archive/bsh_aug_txt/frame/`，其中 `frame_3_aug.*` 為水平翻轉，`frame_4_aug.*` 為垂直翻轉。用 `python flip/demo_API.py --help` 查看自訂輸入、輸出與變換選項。

若要改成水平翻轉與順時針旋轉 90 度，並另存結果：

```bash
python flip/demo_API.py --transforms hflip rotate90 --output-dir Archive/bsh_custom
```

## 將 XML 轉為 YOLO TXT

將原圖與同名 `.xml` 放入 `Archive/bsh_aug/`，並在該目錄建立 `classes.txt`。每行填入一個 XML 類別名稱，第一行對應 ID 0。以下指令會將原圖、YOLO 標註與類別清單寫入 `Archive/bsh_aug_txt/`，供上方的擴增指令使用。

```bash
python flip/demo_API.py --convert-only --input Archive/bsh_aug \
  --classes-file Archive/bsh_aug/classes.txt --output-dir Archive/bsh_aug_txt
```

使用完畢後執行 `deactivate` 離開虛擬環境。
