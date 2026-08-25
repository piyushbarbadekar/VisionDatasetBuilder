from pathlib import Path
import cv2


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".ts",
}


class FrameExtractor:
    """
    Extract frames from video files at a fixed time interval.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        interval_seconds=2.0,
        image_extension=".jpg",
        jpeg_quality=95,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.interval_seconds = interval_seconds
        self.image_extension = image_extension
        self.jpeg_quality = jpeg_quality

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def find_videos(self):
        """Return all supported videos in the input directory."""

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        videos = [
            path
            for path in self.input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

        return sorted(videos)

    def extract_video(self, video_path):
        """
        Extract frames from a single video.

        Returns:
            int: Number of frames extracted.
        """

        video_path = Path(video_path)

        video_name = video_path.stem
        output_folder = self.output_dir / video_name

        # Skip videos that have already been processed.
        if output_folder.exists() and any(output_folder.iterdir()):
            print(f"[SKIP] {video_name} already processed.")
            return 0

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path.name}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            print(f"[ERROR] Invalid FPS for {video_path.name}")
            cap.release()
            return 0

        frame_interval = max(
            1,
            int(fps * self.interval_seconds)
        )

        extracted = 0
        frame_idx = 0

        print(f"\nProcessing: {video_name}")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % frame_interval == 0:

                filename = (
                    output_folder
                    / f"frame_{frame_idx:06d}{self.image_extension}"
                )

                cv2.imwrite(
                    str(filename),
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        self.jpeg_quality,
                    ],
                )

                extracted += 1

            frame_idx += 1

            if total_frames > 0 and frame_idx % 500 == 0:
                progress = (
                    frame_idx / total_frames
                ) * 100

                print(
                    f"\rProgress: {progress:5.1f}%",
                    end=""
                )

        cap.release()

        print("\rProgress: 100.0%")
        print(f"Frames extracted: {extracted}")

        return extracted

    def run(self):
        """Extract frames from all videos in the input directory."""

        videos = self.find_videos()

        if not videos:
            print(
                f"No supported videos found in "
                f"{self.input_dir}"
            )
            return

        total_frames = 0

        print("=" * 60)
        print("FRAME EXTRACTION")
        print("=" * 60)

        for video in videos:
            total_frames += self.extract_video(video)

        print("\n" + "=" * 60)
        print("FRAME EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Videos found: {len(videos)}")
        print(f"Total frames extracted: {total_frames}")
        print("=" * 60)