# Notch-Tech-AI-Task

# Object Detection and Caption Generation Pipeline

This project combines state-of-the-art computer vision and natural language processing (NLP) models to create a pipeline for:

1. Detecting objects in images using YOLOv8.
2. Generating descriptive captions based on detected objects using FLAN-T5.
3. Displaying images alongside their generated captions.
4. Saving results to a JSON file for further analysis.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8+
- Required libraries:
  ```bash
  pip install torch transformers pillow ultralytics ensemble-boxes matplotlib
  ```
- Access to a GPU (optional but recommended for faster inference).

## Models Used

### YOLOv8
Used for high-accuracy object detection.
- Pre-trained model: `yolov8x6.pt`

### FLAN-T5
A fine-tuned model for language generation tasks, used here to create image captions.
- Model name: `google/flan-t5-large`

## How It Works

### 1. Object Detection
The pipeline uses YOLOv8 to detect objects in an image. Detected bounding boxes and labels are processed with the Weighted Boxes Fusion (WBF) technique to improve accuracy by consolidating overlapping predictions.

### 2. Caption Generation
Using detected object labels, a descriptive caption is generated with FLAN-T5 by creating a structured prompt.

### 3. Visualization
The image, along with the generated caption, is displayed for easy verification.

### 4. Export Results
Captions, along with detected objects, are saved in a JSON file for further analysis.

## Pipeline Functions

### `detect_objects`
Detects objects in an image using YOLOv8.
- Inputs: Image path, confidence threshold, IoU threshold, image size.
- Output: List of detected object labels.

### `generate_caption_with_llm`
Generates a descriptive caption based on detected labels.
- Inputs: List of detected object labels.
- Output: Textual description.

### `process_images`
Processes all images in a specified folder.
- Inputs: Folder path, output JSON file path.
- Output: Saves detected objects and captions in JSON format.

## Usage

### Mount Google Drive (Optional)
If using Google Colab, mount Google Drive to access images:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Main Script
Specify the folder containing images and the output JSON file name, then run:
```python
if __name__ == "__main__":
    image_folder = "/content/drive/MyDrive/test images"  # Update with your folder path
    output_file = "captions_with_factual_descriptions.json"
    process_images(image_folder, output_file)
```

## Example Output
Each entry in the JSON file includes:
- Image name.
- Generated caption.
- List of detected objects.

Example:
```json
[
    {
        "image": "example.jpg",
        "caption": "A dog sitting on a grassy field with a ball nearby.",
        "objects": ["dog", "ball", "grass"]
    }
]
```

## Visualization
Images are displayed with captions for verification:
- Detected objects are overlaid on the image.
- Captions provide a descriptive summary.

## Notes
- Ensure your images are in `.jpg`, `.jpeg`, or `.png` format.
- Adjust the `confidence` and `iou` thresholds in `detect_objects` for better results based on your dataset.
- GPU is recommended for faster inference, but the code will work on CPU as well.

## Acknowledgments
This project leverages open-source models and libraries:
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
