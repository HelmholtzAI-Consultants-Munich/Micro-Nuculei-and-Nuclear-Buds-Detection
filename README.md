# Micro-Nuclei and Nuclear Buds Detection

A [napari](https://napari.org) plugin for detecting and annotating micro-nuclei and nuclear buds in biological microscopy images. This tool combines automated nuclei segmentation with manual annotation capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-napari-orange)

## Features

- **Data Management**: Browse and navigate through image datasets with hierarchical folder support
- **Nuclei Segmentation**: Automated nuclei segmentation using [Cellpose](https://www.cellpose.org/) with adjustable parameters
- **Manual Annotation**: Interactive bounding box annotation for micro-nuclei and nuclear buds
- **YOLO Format Export**: Annotations are saved in YOLO format for compatibility with object detection frameworks
- **Postprocessing**: Automatic assignment of detections to the nearest segmented nucleus
- **Statistics Generation**: Statistics of detection counts

## Installation

### Prerequisites

- conda or miniconda installation

### Install from Source

Run the following commands in the terminall (or in Anaconda prompt for Windows).

1. Clone the repository:

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/Micro-Nuculei-and-Nuclear-Buds-Detection.git
cd Micro-Nuculei-and-Nuclear-Buds-Detection
```

2. Create and activate a conda virtual environment:

```bash
conda create -n buds python=3.12
conda activate buds
```

3. Install the package in development mode:

```bash
pip install -e .
```

This will install all required dependencies including:
- `napari[all]>=0.4.0`
- `cellpose==3.1.1.1`
- `numpy`
- `scikit-image`
- `qtpy`

## Usage

To stat the application run the following commands in the terminall (or in Anaconda prompt for Windows).

1. Activate your conda environment:

```bash
conda activate buds
```

2. Launch napari with all widgets pre-loaded using the command-line tool:

```bash
mb-detect
```

### Workflow

#### 1. Load a Dataset

1. In the **Data Management** widget, click **Browse** to select your dataset folder
2. The folder tree will display all `.tif` images found in the directory (including subdirectories)
3. Double-click an image to load it into the viewer

#### 2. Segment Nuclei

Nuclei segmentation runs automatically when an image is loaded. To customize the segmentation, use the **Nuclei Segmentation** widget to adjust parameters:

- **Min Area**: Minimum nucleus area in pixels
- **Cell Probability Threshold**: Cellpose confidence threshold
- **Diameter**: Expected nucleus diameter (0 for auto-detection)

After changing parameters, click **Segment Nuclei** to re-run the segmentation with the new settings.

#### 3. Annotate Detections

1. Use the **Micro-Nuclei Detection** widget
2. Select the annotation class (Micro-Nuclei or Nuclear Buds)
3. Draw bounding boxes around detected structures
4. Use **Select Shapes** to modify or delete annotations
5. Use **Move Camera** to pan/zoom without drawing

#### 4. Save

Annotations are saved automatically when:
- The **Save** button is clicked
- The next image is loaded

Assignation of detectios to the respecive nuceli is happens automatically when annotations are saved.

### Output Files

The plugin creates the following folder structure in your dataset directory:

```
dataset/
├── annotations/           # YOLO format annotation files
│   └── *.txt
├── nuclei_segmentation/   # Segmentation masks
│   ├── *.npy
│   └── nuclei_segmentation_params.json
└── postprocessing/        # Detection statistics
    ├── *.csv              # Per-image results
    └── summary_statistics.csv
```

## Annotation Format

Annotations are stored in YOLO format:

```
<class_id> <center_x> <center_y> <width> <height>
```

Where:
- `class_id`: 0 for micro-nuclei, 1 for nuclear buds
- All coordinates are normalized to [0, 1] range

## Postprocessing Output Format

### Per-Image Files (`*.csv`)

Each image generates a tab-separated file with detection counts per nucleus:

| Column | Description |
|--------|-------------|
| `nucleus_id` | Unique identifier for each segmented nucleus |
| `micro_nuclei` | Number of micro-nuclei assigned to this nucleus |
| `nuclear_buds` | Number of nuclear buds assigned to this nucleus |

### Summary Statistics (`summary_statistics.csv`)

A single file aggregating statistics across all processed images:

| Column | Description |
|--------|-------------|
| `image_id` | Relative path to the image file |
| `micro_nuclei_count` | Total micro-nuclei detected in the image |
| `nuclear_buds_count` | Total nuclear buds detected in the image |
| `nuclei_count` | Total number of segmented nuclei |
| `1_micro_nuclei` | Fraction of nuclei with exactly 1 micro-nucleus |
| `1_nuclear_buds` | Fraction of nuclei with exactly 1 nuclear bud |
| `2_micro_nuclei` | Fraction of nuclei with exactly 2 micro-nuclei |
| `2_nuclear_buds` | Fraction of nuclei with exactly 2 nuclear buds |
| `3_micro_nuclei` | Fraction of nuclei with exactly 3 micro-nuclei |
| `3_nuclear_buds` | Fraction of nuclei with exactly 3 nuclear buds |
| `4+_micro_nuclei` | Fraction of nuclei with 4 or more micro-nuclei |
| `4+_nuclear_buds` | Fraction of nuclei with 4 or more nuclear buds |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
