"""Augment object-detection images with YOLO TXT or Pascal VOC XML labels."""

import argparse
import filecmp
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import cv2
import imgaug as ia
from imgaug import augmenters as iaa


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _original_images(images):
    """Skip numbered variants only when their source is present in the same directory."""
    names = {p.name for p in images}
    originals = []
    for image in images:
        match = re.fullmatch(r"(.+)_([1-9][0-9]*)_aug", image.stem)
        if match and int(match[2]) >= 3 and match[1] + image.suffix in names:
            continue
        originals.append(image)
    return originals


def _file_id(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _read_txt(path, width, height, classes):
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_id, *values = line.split()
            label = int(class_id)
            x, y, w, h = map(float, values)
            if label < 0 or (classes and label >= len(classes)):
                raise ValueError(f"unknown class ID {label}")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("expected normalized center and positive width/height")
            left, right = x - w / 2, x + w / 2
            top, bottom = y - h / 2, y + h / 2
            # Rounding a YOLO row to six decimals can move an edge just outside.
            if min(left, top) < -1e-6 or max(right, bottom) > 1 + 1e-6:
                raise ValueError("box extends outside the image")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        boxes.append(ia.BoundingBox(max(0, left) * width, max(0, top) * height,
                                    min(1, right) * width, min(1, bottom) * height,
                                    label=label))
    return boxes


def _read_xml(path, width, height):
    boxes = []
    try:
        root = ET.parse(path).getroot()
        size = root.find("size")
        if size is not None:
            if (int(size.findtext("width", "")), int(size.findtext("height", ""))) != (width, height):
                raise ValueError(f"XML size differs from the image ({width} x {height})")
        for obj in root.findall("object"):
            label = obj.findtext("name", "").strip()
            coords = [float(obj.findtext(f"bndbox/{name}", ""))
                      for name in ("xmin", "ymin", "xmax", "ymax")]
            x1, y1, x2, y2 = coords
            if not label or not all(map(math.isfinite, coords)) or x1 >= x2 or y1 >= y2:
                raise ValueError(f"invalid box for {label!r}: {coords}")
            boxes.append(ia.BoundingBox(*coords, label=label))
    except (ET.ParseError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return boxes


def _visible_boxes(boxes, width, height):
    for box in boxes.bounding_boxes:
        x1, y1 = max(0, box.x1), max(0, box.y1)
        x2, y2 = min(width, box.x2), min(height, box.y2)
        if x2 > x1 and y2 > y1:
            yield box.label, x1, y1, x2, y2


def _yolo_text(boxes, width, height):
    lines = []
    for label, x1, y1, x2, y2 in _visible_boxes(boxes, width, height):
        x, y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{label} {x:.9g} {y:.9g} {w:.9g} {h:.9g}\n")
    return "".join(lines)


def _already_installed(target, content):
    if not target.exists():
        return False
    if isinstance(content, Path):
        matches = filecmp.cmp(content, target, shallow=False)
    else:
        matches = target.read_text(encoding="utf-8-sig") == content
    if not matches:
        raise FileExistsError(f"Destination already contains a different file: {target}")
    return True


def _install_new(target, content):
    """Publish a complete conversion output without replacing an existing file."""
    if _already_installed(target, content):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as scratch:
        staged = Path(scratch) / target.name
        if isinstance(content, Path):
            shutil.copyfile(content, staged)
        else:
            staged.write_text(content, encoding="utf-8")
        try:
            target.hardlink_to(staged)
        except FileExistsError:
            _already_installed(target, content)
        except OSError:
            # Some removable/network filesystems cannot create hard links.
            # Reserve the name exclusively before replacing it on these volumes.
            try:
                target.touch(exist_ok=False)
            except FileExistsError:
                _already_installed(target, content)
            else:
                try:
                    staged.replace(target)
                except OSError:
                    target.unlink(missing_ok=True)
                    raise


class Augmentor:
    """Use an image directory or a manifest with one image path per line.

    TXT stores zero-based class IDs and normalized center x, center y, width,
    height. Each augmenter creates one variant per image. Pass an iaa.Sequential
    to compose several operations into one variant. Relative manifest entries
    resolve against the manifest directory, then the working directory.
    """

    def __init__(self, input_file=None, *, input_dir=None, output_dir=None,
                 label_format=None, output_format=None, labels_dir=None,
                 classes_file=None, seed=1):
        if input_file is not None and input_dir is not None:
            raise ValueError("Choose either input_file or input_dir")
        source = Path(input_dir or input_file or "train.txt").expanduser().resolve()
        self.manifest = None
        if input_dir is not None or source.is_dir():
            paths = sorted(p for p in source.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            self.paths = _original_images(paths)
        else:
            self.manifest = source
            paths = []
            for entry in source.read_text(encoding="utf-8-sig").splitlines():
                if entry.strip():
                    path = Path(entry.strip()).expanduser()
                    relative = source.parent / path
                    paths.append((relative if relative.exists() else path).resolve())
            self.paths = paths
        if not paths:
            raise ValueError(f"No images found in {source}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")

        self._all_paths = paths
        self.labels_dir = Path(labels_dir).resolve() if labels_dir is not None else None
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        if label_format is None:
            label_format = "txt" if self._label_path(self.paths[0], "txt").exists() else "xml"
        self.label_format = label_format
        self.output_format = output_format or label_format
        if self.label_format not in ("txt", "xml") or self.output_format not in ("txt", "xml"):
            raise ValueError("Label formats must be 'txt' or 'xml'")
        if len({(self._output_root(path), path.stem.casefold()) for path in paths}) != len(paths):
            raise ValueError("Input images would produce duplicate output names")

        self.classes_file = Path(classes_file).resolve() if classes_file is not None else None
        if self.classes_file is None:
            roots = {self._label_path(path, self.label_format).parent for path in paths}
            if len(roots) == 1:
                candidate = roots.pop() / "classes.txt"
                if candidate.is_file():
                    self.classes_file = candidate
        self.classes = []
        if self.classes_file is not None:
            self.classes = [line.strip() for line in
                            self.classes_file.read_text(encoding="utf-8-sig").splitlines()]
            if not self.classes or not all(self.classes) or len(set(self.classes)) != len(self.classes):
                raise ValueError(f"Expected one unique class name per line in {self.classes_file}")
        if self.label_format != self.output_format and not self.classes:
            raise ValueError("Changing label formats requires a classes.txt file in class-ID order")
        self.class_ids = {name: index for index, name in enumerate(self.classes)}
        ia.seed(seed)
        self.aug = []
        self.images = []
        self.bbs = []

    def _label_path(self, image, fmt):
        if self.labels_dir is not None:
            return self.labels_dir / image.with_suffix("." + fmt).name
        if fmt == "xml" and "JPEGImages" in image.parts:
            parts = list(image.parts)
            parts[parts.index("JPEGImages")] = "Annotations"
            return Path(*parts).with_suffix(".xml")
        return image.with_suffix("." + fmt)

    def _output_root(self, image):
        if self.output_dir is not None:
            return self.output_dir
        if self.label_format == "xml" and "JPEGImages" in image.parts:
            parts = list(image.parent.parts)
            parts[parts.index("JPEGImages")] = "Augmented_dataset"
            return Path(*parts)
        return image.parent

    def readImages(self):
        """Load each image and its boxes, including images with empty labels."""
        self.images, self.bbs = self._load_images(self.paths)

    def _load_images(self, paths):
        images, annotations = [], []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode image: {path}")
            height, width = image.shape[:2]
            label_path = self._label_path(path, self.label_format)
            if self.label_format == "txt":
                boxes = _read_txt(label_path, width, height, self.classes)
                if self.output_format == "xml":
                    for box in boxes:
                        box.label = self.classes[box.label]
            else:
                boxes = _read_xml(label_path, width, height)
                if self.output_format == "txt":
                    for box in boxes:
                        if box.label not in self.class_ids:
                            raise ValueError(f"{label_path}: unknown XML class {box.label!r}")
                        box.label = self.class_ids[box.label]
            images.append(image)
            annotations.append(ia.BoundingBoxesOnImage(boxes, shape=image.shape))
        return images, annotations

    def addAugmentor(self, augmentor):
        """Add one imgaug augmenter (or a Sequential pipeline) as a variant."""
        self.aug.append(augmentor)

    def _copy_classes(self, roots):
        if self.output_format == "txt" and self.classes_file is not None:
            for root in roots:
                _install_new(root / "classes.txt", self.classes_file)

    def convert_to_txt(self):
        """Convert original XML labels beside byte-identical copies of the images."""
        if (self.label_format, self.output_format) != ("xml", "txt"):
            raise ValueError("Conversion needs label_format='xml' and output_format='txt'")
        # Existing variants with their own XML are valid conversion inputs too.
        originals = set(self.paths)
        paths = [p for p in self._all_paths
                 if p in originals or self._label_path(p, "xml").is_file()]
        if paths == self.paths and self.images:
            images, annotations = self.images, self.bbs
        else:
            images, annotations = self._load_images(paths)
        files = []
        for path, image, boxes in zip(paths, images, annotations):
            out_image = (self.output_dir or path.parent) / path.name
            target = out_image.with_suffix(".txt")
            if target.name == "classes.txt":
                raise ValueError(f"Label name conflicts with the class list: {target}")
            height, width = image.shape[:2]
            text = _yolo_text(boxes, width, height)
            _already_installed(out_image, path)
            _already_installed(target, text)
            files.append((path, out_image, target, text))

        self._copy_classes({target.parent for _, _, target, _ in files})
        for path, out_image, target, text in files:
            _install_new(out_image, path)
            _install_new(target, text)
        return len(files)

    def _variants(self):
        """Plan flat output names and detect collisions before writing any files."""
        inputs = set(self.paths)
        inputs.update(self._label_path(p, self.label_format) for p in self.paths)
        inputs.update(p for p in (self.manifest, self.classes_file) if p is not None)
        input_ids = {_file_id(p) for p in inputs}
        variants = []
        for index, transform in enumerate(self.aug, 3):
            for path, image, boxes in zip(self.paths, self.images, self.bbs):
                folder = self._output_root(path)
                stem = f"{path.stem}_{index}_aug"
                out_image = folder / (stem + path.suffix)
                out_label = folder / (stem + "." + self.output_format)
                for target in (out_image, out_label):
                    if target.exists() and _file_id(target) in input_ids:
                        raise ValueError(f"Output would overwrite an input: {target}. Choose another output directory.")
                variants.append((transform, image, boxes, out_image, out_label))
        return variants

    def augment(self):
        """Write image/label pairs directly in the output directory and return the count."""
        if not self.images:
            self.readImages()
        variants = self._variants()
        self._copy_classes({self._output_root(path) for path in self.paths})
        for transform, image, boxes, out_image, out_label in variants:
            # One imgaug call shares all random choices across the image and its boxes.
            result, transformed = transform(image=image, bounding_boxes=boxes)
            height, width = result.shape[:2]
            out_image.parent.mkdir(parents=True, exist_ok=True)
            # Finish encoding both files before replacing a previous variant.
            with TemporaryDirectory(dir=out_image.parent) as scratch:
                tmp_image = Path(scratch) / out_image.name
                tmp_label = Path(scratch) / out_label.name
                if not cv2.imwrite(str(tmp_image), result):
                    raise OSError(f"Cannot write augmented image: {out_image}")
                if self.output_format == "txt":
                    tmp_label.write_text(_yolo_text(transformed, width, height), encoding="utf-8")
                else:
                    from pascal_voc_writer import Writer
                    writer = Writer(str(out_image), width, height)
                    for name, x1, y1, x2, y2 in _visible_boxes(transformed, width, height):
                        if int(x2) > int(x1) and int(y2) > int(y1):
                            writer.addObject(name, int(x1), int(y1), int(x2), int(y2))
                    writer.save(str(tmp_label))
                tmp_image.replace(out_image)
                tmp_label.replace(out_label)
        return len(variants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply grayscale, sharpen and brightness augmentation.")
    parser.add_argument("--input", required=True, help="Image directory or manifest file")
    parser.add_argument("--output-dir", type=Path, help="Destination for these variants")
    parser.add_argument("--num", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    augmentor = Augmentor(args.input, output_dir=args.output_dir)
    transforms = [iaa.Grayscale(alpha=(0.5, 1.0)), iaa.Sharpen(alpha=(0, 0.25), lightness=(0.75, 1.5)),
                  iaa.Multiply((0.75, 1.25))]
    for transform in transforms[:args.num]:
        augmentor.addAugmentor(transform)
    print(f"Generated {augmentor.augment()} image/label pairs.")
