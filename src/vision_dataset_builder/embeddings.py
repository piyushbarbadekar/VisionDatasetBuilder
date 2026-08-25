from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


class CLIPEmbedder:
    """
    Generate normalized CLIP image embeddings.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        model_name="openai/clip-vit-base-patch32",
        batch_size=64,
        device=None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.model_name = model_name
        self.batch_size = batch_size

        # Automatically select the best available device.
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {self.model_name}")

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model = CLIPModel.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print("CLIP model loaded.")

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

    def _load_image(self, image_path):
        """Safely load an image as RGB."""

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")

        except Exception as error:
            print(
                f"\n[WARNING] Could not load "
                f"{image_path}: {error}"
            )
            return None

    def generate(self):
        """
        Generate embeddings for all images.

        Returns:
            embeddings: NumPy array
            filenames: list of image paths
        """

        image_paths = self.find_images()

        if not image_paths:
            print(
                f"No supported images found in "
                f"{self.input_dir}"
            )
            return None, []

        print("=" * 60)
        print("CLIP EMBEDDING GENERATION")
        print("=" * 60)
        print(f"Images found: {len(image_paths)}")

        all_embeddings = []
        valid_filenames = []

        for start in tqdm(
            range(
                0,
                len(image_paths),
                self.batch_size
            )
        ):

            batch_paths = image_paths[
                start:start + self.batch_size
            ]

            images = []
            batch_filenames = []

            for path in batch_paths:

                image = self._load_image(path)

                if image is None:
                    continue

                images.append(image)
                batch_filenames.append(
                    str(path)
                )

            if not images:
                continue

            try:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():

                    outputs = self.model.get_image_features(
                         **inputs
                    )

                    if hasattr(outputs, "pooler_output"):
                        features = outputs.pooler_output
                    else:
                        features = outputs

                    features = (
                        features
                        / features.norm(
                        dim=-1,
                        keepdim=True
                            )
                    )

                all_embeddings.append(
                    features.cpu().numpy()
                )

                valid_filenames.extend(
                    batch_filenames
                )

            except Exception as error:

                print(
                    f"\n[WARNING] Failed to process "
                    f"batch starting at {start}: {error}"
                )

        if not all_embeddings:
            print("No embeddings were generated.")
            return None, []

        embeddings = np.concatenate(
            all_embeddings,
            axis=0
        )

        self.save(
            embeddings,
            valid_filenames
        )

        print("\n" + "=" * 60)
        print("CLIP EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"Images processed: {len(valid_filenames)}")
        print(f"Embedding shape: {embeddings.shape}")
        print("=" * 60)

        return embeddings, valid_filenames

    def save(self, embeddings, filenames):
        """Save embeddings and corresponding filenames."""

        embedding_path = (
            self.output_dir / "embeddings.npy"
        )

        filename_path = (
            self.output_dir / "filenames.json"
        )

        np.save(
            embedding_path,
            embeddings
        )

        with open(
            filename_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                filenames,
                file,
                indent=4
            )

        print(f"Embeddings saved: {embedding_path}")
        print(f"Filenames saved: {filename_path}")