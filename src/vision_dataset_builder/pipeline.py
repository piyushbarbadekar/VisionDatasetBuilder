from pathlib import Path

import yaml

from .extraction import FrameExtractor
from .detection import ObjectDetector
from .embeddings import CLIPEmbedder
from .clustering import EmbeddingClusterer
from .sampling import RepresentativeSampler
from .quality import DatasetQualityAnalyzer
from .dataset import YOLODatasetBuilder


class VisionDatasetPipeline:
    """
    Orchestrates the complete VisionDatasetBuilder pipeline.
    """

    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = yaml.safe_load(file)

        self.project_root = (
            self.config_path.parent.parent.resolve()
        )

    def path(self, relative_path):
        """Convert a project-relative path to an absolute path."""

        return self.project_root / relative_path

    def run_extraction(self):
        """Run video frame extraction."""

        config = self.config

        extractor = FrameExtractor(
            input_dir=self.path(
                config["input"]["video_dir"]
            ),
            output_dir=self.path(
                config["output"]["frame_dir"]
            ),
            interval_seconds=config[
                "extraction"
            ]["interval_seconds"],
            image_extension=config[
                "extraction"
            ]["image_extension"],
            jpeg_quality=config[
                "extraction"
            ]["jpeg_quality"],
        )

        extractor.run()

    def run_detection(self):
        """Run YOLO object detection."""

        config = self.config

        detector = ObjectDetector(
            model_path=self.path(
                config["detection"]["model"]
            ),
            input_dir=self.path(
                config["output"]["frame_dir"]
            ),
            output_dir=self.path(
                config["output"]["crop_dir"]
            ),
            confidence=config[
                "detection"
            ]["confidence"],
            min_width=config[
                "detection"
            ]["min_width"],
            min_height=config[
                "detection"
            ]["min_height"],
        )

        detector.run()

    def run_quality_analysis(self):
        """Analyze the generated object crops."""

        config = self.config

        analyzer = DatasetQualityAnalyzer(
            input_dir=self.path(
                config["output"]["crop_dir"]
            ),
            output_file=self.path(
                config["output"]["report_dir"]
            ) / "dataset_quality.csv",
        )

        analyzer.analyze()

    def run_embeddings(self):
        """Generate CLIP embeddings."""

        config = self.config

        embedder = CLIPEmbedder(
            input_dir=self.path(
                config["output"]["crop_dir"]
            ),
            output_dir=self.path(
                config["output"]["embedding_dir"]
            ),
            model_name=config[
                "embedding"
            ]["model"],
            batch_size=config[
                "embedding"
            ]["batch_size"],
            device=config[
            "embedding"
            ]["device"],
        )


        embedder.generate()

    def run_clustering(self):
        """Cluster CLIP embeddings."""

        config = self.config

        clusterer = EmbeddingClusterer(
            embedding_file=self.path(
                config["output"]["embedding_dir"]
            ) / "embeddings.npy",
            filename_file=self.path(
                config["output"]["embedding_dir"]
            ) / "filenames.json",
            output_dir=self.path(
                config["output"]["clustering_dir"]
            ),
            n_clusters=config[
                "clustering"
            ]["clusters"],
            batch_size=config[
                "clustering"
            ]["batch_size"],
            random_state=config[
                "clustering"
            ]["random_state"],
        )

        clusterer.run()

    def run_sampling(self):
        """Copy representative images."""

        config = self.config

        sampler = RepresentativeSampler(
            mapping_file=self.path(
                config["output"]["clustering_dir"]
            ) / "representative_mapping.csv",
            source_dir=self.path(
                config["output"]["crop_dir"]
            ),
            output_dir=self.path(
                config["output"]["representative_dir"]
            ),
        )

        sampler.run()

    def run_dataset_build(self):
        """Build the final YOLO dataset."""

        config = self.config

        class_names = config[
            "dataset"
        ]["classes"]

        builder = YOLODatasetBuilder(
            image_dir=self.path(
                config["output"]["representative_dir"]
            ),
            label_dir=self.path(
                "data/labels"
            ),
            output_dir=self.path(
                config["output"]["dataset_dir"]
            ),
            class_names=class_names,
            train_ratio=config[
                "dataset"
            ]["train_ratio"],
            random_seed=config[
                "dataset"
            ]["random_seed"],
        )

        builder.run()

    def run(self, stages=None):
        """
        Run selected pipeline stages.

        If stages is None, run all stages that do not require
        manual annotation.
        """

        if stages is None:
            stages = [
                "extract",
                "detect",
                "quality",
                "embed",
                "cluster",
                "sample",
            ]

        stage_map = {
            "extract": self.run_extraction,
            "detect": self.run_detection,
            "quality": self.run_quality_analysis,
            "embed": self.run_embeddings,
            "cluster": self.run_clustering,
            "sample": self.run_sampling,
            "dataset": self.run_dataset_build,
        }

        for stage in stages:

            if stage not in stage_map:
                raise ValueError(
                    f"Unknown pipeline stage: {stage}"
                )

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"RUNNING STAGE: {stage.upper()}"
            )

            print(
                "=" * 70
            )

            stage_map[stage]()

            print(
                f"\nSTAGE COMPLETE: {stage.upper()}"
            )