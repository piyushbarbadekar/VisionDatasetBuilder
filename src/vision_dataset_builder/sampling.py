from pathlib import Path
import json
import shutil


class RepresentativeSampler:
    """
    Copy representative images selected by the clustering stage
    into a dedicated directory for annotation.
    """

    def __init__(
        self,
        mapping_file,
        source_dir,
        output_dir,
    ):
        self.mapping_file = Path(mapping_file)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_mapping(self):
        """Load representative image information."""

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping file not found: "
                f"{self.mapping_file}"
            )

        import pandas as pd

        mapping = pd.read_csv(
            self.mapping_file
        )

        if "representative" not in mapping.columns:
            raise ValueError(
                "Mapping file must contain a "
                "'representative' column."
            )

        return mapping

    def find_image(self, filename):
        """
        Find a representative image inside the source directory.
        """

        filename = Path(filename)

        # First check the source directory directly.
        direct_path = (
            self.source_dir / filename
        )

        if direct_path.exists():
            return direct_path

        # Then search recursively.
        matches = list(
            self.source_dir.rglob(
                filename.name
            )
        )

        if matches:
            return matches[0]

        return None

    def copy_representatives(self, mapping):
        """Copy representative images."""

        copied = 0
        missing = []

        for _, row in mapping.iterrows():

            filename = row["representative"]

            source = self.find_image(
                filename
            )

            if source is None:
                missing.append(filename)
                continue

            destination = (
                self.output_dir
                / source.name
            )

            shutil.copy2(
                source,
                destination
            )

            copied += 1

        return copied, missing

    def run(self):
        """Run representative image selection."""

        print("=" * 60)
        print("REPRESENTATIVE IMAGE SAMPLING")
        print("=" * 60)

        mapping = self.load_mapping()

        print(
            f"Representatives requested: "
            f"{len(mapping)}"
        )

        copied, missing = (
            self.copy_representatives(
                mapping
            )
        )

        print("\n" + "=" * 60)
        print("SAMPLING COMPLETE")
        print("=" * 60)
        print(
            f"Images copied: {copied}"
        )
        print(
            f"Images missing: {len(missing)}"
        )

        if missing:
            print("\nMissing images:")
            for filename in missing:
                print(f"  - {filename}")

        print(
            f"\nOutput directory: "
            f"{self.output_dir}"
        )

        print("=" * 60)