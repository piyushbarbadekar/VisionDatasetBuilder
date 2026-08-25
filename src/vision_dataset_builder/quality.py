from pathlib import Path
import csv

import cv2


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


class DatasetQualityAnalyzer:
    """
    Analyze basic image quality without modifying or deleting
    any images.
    """

    def __init__(
        self,
        input_dir,
        output_file,
        max_images=None,
    ):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.max_images = max_images

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def find_images(self):
        """Find supported images recursively."""

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{self.input_dir}"
            )

        images = [
            path
            for path in self.input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        ]

        images = sorted(images)

        if self.max_images is not None:
            images = images[:self.max_images]

        return images

    @staticmethod
    def calculate_sharpness(image):
        """
        Estimate image sharpness using the variance
        of the Laplacian.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    @staticmethod
    def calculate_brightness(image):
        """Calculate average pixel brightness."""

        return float(image.mean())

    def analyze(self):
        """Analyze all selected images."""

        images = self.find_images()

        if not images:
            print(
                f"No supported images found in "
                f"{self.input_dir}"
            )
            return

        print("=" * 60)
        print("DATASET QUALITY ANALYSIS")
        print("=" * 60)
        print(f"Images to analyze: {len(images)}")

        rows = []
        corrupted = 0

        for image_path in images:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                corrupted += 1

                rows.append({
                    "image": str(image_path),
                    "sharpness": "",
                    "brightness": "",
                    "width": "",
                    "height": "",
                    "status": "corrupted",
                })

                continue

            height, width = image.shape[:2]

            sharpness = self.calculate_sharpness(
                image
            )

            brightness = self.calculate_brightness(
                image
            )

            rows.append({
                "image": str(image_path),
                "sharpness": round(
                    sharpness,
                    2
                ),
                "brightness": round(
                    brightness,
                    2
                ),
                "width": width,
                "height": height,
                "status": "valid",
            })

        self.save_report(rows)

        print("\n" + "=" * 60)
        print("QUALITY ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Images analyzed: {len(images)}")
        print(f"Corrupted images: {corrupted}")
        print(f"Report: {self.output_file}")
        print("=" * 60)

    def save_report(self, rows):
        """Save the quality analysis as CSV."""

        fieldnames = [
            "image",
            "sharpness",
            "brightness",
            "width",
            "height",
            "status",
        ]

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)