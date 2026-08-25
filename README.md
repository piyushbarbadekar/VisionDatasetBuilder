# VisionDatasetBuilder

A modular computer-vision pipeline for transforming large image and video datasets into smaller, visually representative datasets for efficient manual annotation.

VisionDatasetBuilder combines video frame extraction, object detection, image-quality analysis, CLIP embeddings, visual clustering, representative sampling, and YOLO dataset preparation into a single configurable pipeline.

---

## Overview

Large computer-vision datasets often contain many visually similar images.

For example, a long CCTV recording can produce thousands of frames, while many of those frames contain nearly identical visual information. Manually annotating every image can therefore be inefficient.

VisionDatasetBuilder addresses this problem by:

1. Extracting frames from videos
2. Detecting objects using YOLO
3. Creating object crops
4. Analyzing basic image quality
5. Generating CLIP visual embeddings
6. Clustering visually similar samples
7. Selecting representative images from each cluster
8. Preparing selected images for manual annotation
9. Building a YOLO-compatible dataset from annotations

The goal is to reduce redundant samples before annotation while maintaining visual diversity.

---

## Pipeline

```text
                    INPUT VIDEO
                         |
                         v
                +------------------+
                | Frame Extraction |
                +--------+---------+
                         |
                         v
                +------------------+
                | Object Detection |
                |      YOLO        |
                +--------+---------+
                         |
                         v
                +------------------+
                |  Object Cropping |
                +--------+---------+
                         |
                         v
                +------------------+
                | Quality Analysis |
                +--------+---------+
                         |
                         v
                +------------------+
                | CLIP Embeddings  |
                +--------+---------+
                         |
                         v
                +------------------+
                |    Clustering    |
                |  MiniBatchKMeans |
                +--------+---------+
                         |
                         v
                +------------------+
                | Representative   |
                |     Sampling     |
                +--------+---------+
                         |
                         v
                +------------------+
                | Manual Annotation|
                +--------+---------+
                         |
                         v
                +------------------+
                |  YOLO Dataset    |
                |     Builder      |
                +------------------+
In other words: fewer redundant images → less annotation → less suffering. 😌
Key Features
Video Frame Extraction

Extract frames from supported video formats at configurable time intervals.

Supported formats include:

MP4
AVI
MOV
MKV
TS

Each video receives its own output directory.

Generic Object Detection

Uses Ultralytics YOLO to detect objects in extracted frames.

The detection stage:

Supports configurable confidence thresholds
Filters detections based on object size
Generates object crops
Records detection metadata
Supports generic YOLO object classes

The pipeline is not tied to a specific application.

Dataset Quality Analysis

Analyzes generated image crops without deleting or modifying the source images.

Current analysis includes:

Sharpness
Brightness
Image width
Image height
Corrupted image detection

Results are saved as a CSV report.

CLIP Embeddings

Generates visual embeddings using:

openai/clip-vit-base-patch32

Embeddings are normalized and stored as NumPy arrays along with their corresponding filenames.

Visual Clustering

Embeddings are grouped using:

MiniBatchKMeans

The image closest to each cluster center is selected as the representative sample.

Representative Sampling

Representative images are copied into a dedicated directory for manual annotation.

For example:

498 object crops
        |
        v
100 visual clusters
        |
        v
100 representative images

The number of clusters is configurable.

YOLO Dataset Preparation

After manual annotation, the selected images and labels can be converted into a YOLO-compatible dataset.
The idea is simple: show the annotator the useful images, not 47 nearly identical frames of someone turning their head 3 degrees.

The dataset stage:

Matches images with labels
Detects missing images
Splits data into training and validation sets
Copies images and labels
Generates data.yaml
Project Structure
VisionDatasetBuilder/
|
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
|
├── configs/
│   └── config.yaml
|
└── src/
    └── vision_dataset_builder/
        ├── __init__.py
        ├── extraction.py
        ├── detection.py
        ├── embeddings.py
        ├── clustering.py
        ├── sampling.py
        ├── quality.py
        ├── dataset.py
        └── pipeline.py
Installation
1. Clone the repository
git clone <repository-url>
cd VisionDatasetBuilder
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
PyTorch and GPU Support

The CLIP embedding stage automatically selects an available device:

NVIDIA GPU with CUDA
        |
        v
      CUDA

Apple Silicon
        |
        v
       MPS

No supported GPU
        |
        v
       CPU

CPU execution is supported, although CLIP embedding generation can be significantly slower.

For NVIDIA systems, install a CUDA-enabled PyTorch build appropriate for the system before running the embedding stage.

YOLO Model

The object detection stage requires YOLO model weights.

The default configuration expects:

models/yolo11s.pt

Create the directory if necessary:

models/

and place the YOLO weights inside it.

The model path can be changed through:

configs/config.yaml

For example:

detection:
  model: models/yolo11s.pt

Model weights are excluded from the Git repository.

Configuration

Pipeline parameters are controlled through:

configs/config.yaml

Example configuration:

input:
  video_dir: data/videos

output:
  frame_dir: data/frames
  crop_dir: data/crops
  embedding_dir: data/embeddings
  clustering_dir: data/clustering
  representative_dir: data/representatives
  report_dir: reports
  dataset_dir: data/yolo_dataset

extraction:
  interval_seconds: 2
  image_extension: .jpg
  jpeg_quality: 95

detection:
  model: models/yolo11s.pt
  confidence: 0.40
  min_width: 20
  min_height: 20

embedding:
  model: openai/clip-vit-base-patch32
  batch_size: 64
  device: auto

clustering:
  method: minibatch_kmeans
  clusters: 100
  batch_size: 1024
  random_state: 42

dataset:
  train_ratio: 0.80
  random_seed: 42

  classes:
    - object
Running the Pipeline
Show available options
python main.py --help
Run the complete pipeline
python main.py
Run selected stages

Multiple stages can be executed together:

python main.py --stages extract detect quality embed cluster sample
Individual Pipeline Stages
1. Frame Extraction
python main.py --stages extract

Extracts frames from the configured video directory.

2. Object Detection
python main.py --stages detect

Runs YOLO detection on extracted frames and generates object crops.

3. Quality Analysis
python main.py --stages quality

Analyzes the generated crops and produces a dataset-quality report.

4. CLIP Embeddings
python main.py --stages embed

Generates normalized CLIP embeddings for the detected object crops.

5. Clustering
python main.py --stages cluster

Clusters the generated embeddings using MiniBatchKMeans and selects cluster representatives.

6. Representative Sampling
python main.py --stages sample

Copies the selected representative images into the representative-image directory.

7. Dataset Building
python main.py --stages dataset

Builds a YOLO-compatible train/validation dataset from manually annotated representative images.

Data Flow

Generated data follows this general structure:

data/
|
├── videos/
|       Input videos
|
├── frames/
|       Extracted video frames
|
├── crops/
|       Detected object crops
|       detections.csv
|
├── embeddings/
|       embeddings.npy
|       filenames.json
|
├── clustering/
|       labels.npy
|       cluster_centers.npy
|       representative_mapping.csv
|       cluster_statistics.csv
|
├── representatives/
|       Selected representative images
|
└── yolo_dataset/
        Final YOLO dataset

Generated data is excluded from version control through .gitignore.

Manual Annotation Workflow

Representative images are intended to be manually annotated before the final dataset-building stage.

Video
  |
  v
Frame Extraction
  |
  v
Object Detection
  |
  v
CLIP Embeddings
  |
  v
Clustering
  |
  v
Representative Images
  |
  v
Manual Annotation
  |
  v
YOLO Labels
  |
  v
Dataset Builder
  |
  v
Train / Validation Dataset

YOLO annotation files use the standard normalized format:

class_id center_x center_y width height
Outputs
Detection Metadata

The detection stage produces:

data/crops/detections.csv

This contains metadata associated with detected objects and generated crops.

Dataset Quality Report

The quality stage produces:

reports/dataset_quality.csv

The report contains measurements such as:

Image name
Sharpness
Brightness
Width
Height
Clustering Results

The clustering stage produces:

data/clustering/labels.npy
data/clustering/cluster_centers.npy
data/clustering/representative_mapping.csv
data/clustering/cluster_statistics.csv
Example Reduction

The primary goal of the pipeline is to reduce redundant samples before annotation.

For example:

10,000 extracted object images
            |
            v
      CLIP embeddings
            |
            v
       500 clusters
            |
            v
   500 representative images

The actual reduction depends on the dataset and clustering configuration.

Design Philosophy
Reduce Redundancy Before Annotation

Large video datasets frequently contain repeated or visually similar samples.

Clustering allows representative samples to be selected before expensive manual annotation.

Keep Application Logic Separate

The core pipeline is designed to avoid application-specific PPE rules or class definitions.

The same methodology can therefore be adapted to different computer-vision problems.

Make Each Stage Independently Usable

Each major stage can be executed separately:

Extraction
Detection
Quality Analysis
Embedding
Clustering
Sampling
Dataset Building

This makes experimentation and debugging easier.

Limitations

The current implementation has several limitations:

Representative selection depends on embedding quality.
KMeans requires the number of clusters to be specified.
Object detection quality affects downstream crops.
Very small or poorly detected objects may be discarded.
CLIP embedding generation can be computationally expensive.
CPU execution is slower than GPU execution.
Manual annotation is still required before supervised training.
Current quality analysis uses basic image-quality metrics.
Duplicate and near-duplicate detection is not currently implemented.
Future Improvements

Potential improvements include:

Automatic cluster-count selection
Duplicate and near-duplicate detection
Advanced image-quality scoring
Active-learning based sampling
Additional embedding models
HDBSCAN clustering
Parallel video processing
GPU optimization
Interactive dataset inspection
Annotation-tool integration
Dataset statistics dashboard
Command-line configuration overrides
Additional annotation formats
Automated dataset balancing
Technologies
Python
OpenCV
Ultralytics YOLO
PyTorch
Hugging Face Transformers
CLIP
NumPy
Pandas
Scikit-learn
PyYAML
tqdm
Development Background

The initial development of this pipeline was motivated by a real-world computer-vision dataset creation problem involving factory CCTV footage.

The generalized implementation removes application-specific assumptions and focuses on the reusable dataset-curation methodology:

Large Visual Dataset
        |
        v
Automated Processing
        |
        v
Visual Representation
        |
        v
Clustering
        |
        v
Representative Sampling
        |
        v
Efficient Annotation

Built with Python, computer vision, caffeine, and an unreasonable number of terminal commands.

License

This project is intended for educational, research, and experimental use.

Add an appropriate open-source license before distributing the repository publicly.
If this saves someone from manually labeling thousands of redundant images, mission accomplished. 🫡
