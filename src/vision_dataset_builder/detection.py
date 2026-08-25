from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


class ObjectDetector:
    """
    Detect objects in images using a YOLO model
    and save cropped detections.
    """

    def __init__(
        self,
        model_path,
        input_dir,
        output_dir,
        confidence=0.40,
        min_width=20,
        min_height=20,
        classes=None,
    ):
        self.model_path = Path(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.confidence = confidence
        self.min_width = min_width
        self.min_height = min_height
        self.classes = classes

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        print("Model loaded.")

    def find_images(self):
        """Find all supported images recursively."""

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        images = [
            path
            for path in self.input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        ]

        return sorted(images)

    def detect_image(self, image_path):
        """
        Detect objects in one image and save valid crops.

        Returns:
            list: Detection metadata.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read {image_path}")
            return []

        results = self.model.predict(
            image,
            conf=self.confidence,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []

        image_output_dir = (
            self.output_dir / image_path.parent.name
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        object_id = 0

        for box in results.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            width = x2 - x1
            height = y2 - y1

            if width < self.min_width:
                continue

            if height < self.min_height:
                continue

            # Keep coordinates inside the image.
            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_name = (
                f"{image_path.stem}"
                f"_obj{object_id:03d}.jpg"
            )

            crop_path = image_output_dir / crop_name

            cv2.imwrite(
                str(crop_path),
                crop
            )

            class_name = self.model.names[class_id]

            detections.append({
                "image": str(image_path),
                "crop": str(crop_path),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
            })

            object_id += 1

        return detections

    def run(self):
        """Process all images and save detection metadata."""

        images = self.find_images()

        if not images:
            print(
                f"No supported images found in "
                f"{self.input_dir}"
            )
            return

        all_detections = []

        print("=" * 60)
        print("OBJECT DETECTION")
        print("=" * 60)
        print(f"Images found: {len(images)}")

        for image_path in images:

            detections = self.detect_image(
                image_path
            )

            all_detections.extend(detections)

        self.save_metadata(all_detections)

        print("\n" + "=" * 60)
        print("OBJECT DETECTION COMPLETE")
        print("=" * 60)
        print(f"Images processed: {len(images)}")
        print(f"Objects detected: {len(all_detections)}")
        print("=" * 60)

    def save_metadata(self, detections):
        """Save detection information to CSV."""

        csv_path = (
            self.output_dir / "detections.csv"
        )

        if not detections:
            return

        fieldnames = [
            "image",
            "crop",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(detections)

        print(f"Metadata saved: {csv_path}")