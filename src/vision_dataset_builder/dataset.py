from pathlib import Path
import random
import shutil


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


class YOLODatasetBuilder:
    """
    Build a YOLO-format dataset from images and YOLO annotations.

    The builder:
    - matches images with their labels
    - shuffles the dataset reproducibly
    - splits it into train/validation sets
    - copies the files into YOLO directory structure
    - generates data.yaml
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        output_dir,
        class_names,
        train_ratio=0.8,
        random_seed=42,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)

        self.class_names = class_names
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        if not 0 < train_ratio < 1:
            raise ValueError(
                "train_ratio must be between 0 and 1."
            )

    def find_images(self):
        """Find all supported images recursively."""

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: "
                f"{self.image_dir}"
            )

        images = [
            path
            for path in self.image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        ]

        return sorted(images)

    def create_pairs(self):
        """
        Match each image with its corresponding YOLO label.

        Returns:
            pairs: list of (image_path, label_path)
            missing_labels: images without labels
        """

        images = self.find_images()

        pairs = []
        missing_labels = []

        for image in images:

            label = (
                self.label_dir
                / f"{image.stem}.txt"
            )

            if label.exists():
                pairs.append(
                    (image, label)
                )
            else:
                missing_labels.append(
                    image.name
                )

        return pairs, missing_labels

    def split_dataset(self, pairs):
        """Split image-label pairs into train and validation sets."""

        random.seed(self.random_seed)

        pairs = pairs.copy()
        random.shuffle(pairs)

        split_index = int(
            len(pairs) * self.train_ratio
        )

        train_pairs = pairs[:split_index]
        val_pairs = pairs[split_index:]

        return train_pairs, val_pairs

    def create_directories(self):
        """Create the YOLO dataset directory structure."""

        directories = [
            self.output_dir
            / "images"
            / "train",

            self.output_dir
            / "images"
            / "val",

            self.output_dir
            / "labels"
            / "train",

            self.output_dir
            / "labels"
            / "val",
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    @staticmethod
    def copy_pairs(
        pairs,
        image_destination,
        label_destination,
    ):
        """Copy images and labels to their destinations."""

        for image, label in pairs:

            shutil.copy2(
                image,
                image_destination
                / image.name
            )

            shutil.copy2(
                label,
                label_destination
                / label.name
            )

    def write_yaml(self):
        """Generate the YOLO data.yaml file."""

        yaml_path = (
            self.output_dir / "data.yaml"
        )

        with open(
            yaml_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                f"path: {self.output_dir.resolve()}\n"
            )

            file.write(
                "train: images/train\n"
            )

            file.write(
                "val: images/val\n\n"
            )

            file.write("names:\n")

            for index, name in enumerate(
                self.class_names
            ):
                file.write(
                    f"  {index}: {name}\n"
                )

        return yaml_path

    def run(self):
        """Build the complete YOLO dataset."""

        print("=" * 60)
        print("YOLO DATASET BUILD")
        print("=" * 60)

        pairs, missing_labels = (
            self.create_pairs()
        )

        print(
            f"Valid image-label pairs: "
            f"{len(pairs)}"
        )

        print(
            f"Images without labels: "
            f"{len(missing_labels)}"
        )

        if not pairs:
            print(
                "No valid image-label pairs found."
            )
            return

        train_pairs, val_pairs = (
            self.split_dataset(pairs)
        )

        print(
            f"Training images: "
            f"{len(train_pairs)}"
        )

        print(
            f"Validation images: "
            f"{len(val_pairs)}"
        )

        self.create_directories()

        self.copy_pairs(
            train_pairs,
            self.output_dir
            / "images"
            / "train",
            self.output_dir
            / "labels"
            / "train",
        )

        self.copy_pairs(
            val_pairs,
            self.output_dir
            / "images"
            / "val",
            self.output_dir
            / "labels"
            / "val",
        )

        yaml_path = self.write_yaml()

        print("\n" + "=" * 60)
        print("YOLO DATASET BUILD COMPLETE")
        print("=" * 60)

        print(
            f"Dataset location: "
            f"{self.output_dir}"
        )

        print(
            f"data.yaml: "
            f"{yaml_path}"
        )

        print("=" * 60)