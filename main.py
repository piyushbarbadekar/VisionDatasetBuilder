import argparse

from src.vision_dataset_builder.pipeline import VisionDatasetPipeline


def main():
    parser = argparse.ArgumentParser(
        description="VisionDatasetBuilder"
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "extract",
            "detect",
            "quality",
            "embed",
            "cluster",
            "sample",
            "dataset",
        ],
        help="Pipeline stages to run",
    )

    args = parser.parse_args()

    pipeline = VisionDatasetPipeline(
        args.config
    )

    pipeline.run(args.stages)


if __name__ == "__main__":
    main()