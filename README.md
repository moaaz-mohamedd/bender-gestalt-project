# AI-Based Analysis of Bender Gestalt Test Drawings

## Cognitive Science Course Project

This project is an academic AI prototype for analyzing **Bender Gestalt Test** drawings using **Computer Vision** and **Machine Learning**.

The system converts scanned hand-drawn images into structured numerical features, applies skeleton-based image analysis, compares drawings with reference templates, and trains machine learning models to predict the final pathological classification from the provided Excel dataset.

---

## Project Overview

The **Bender Gestalt Test** is a neuropsychological assessment where a subject copies a set of geometric figures.

The copied drawings may reflect:

- Visual-motor integration
- Perceptual organization
- Distortion
- Perseveration
- Rotation
- Poor closure
- Integration failure
- Spatial disorganization

This project focuses on two main parts:

1. **Computer Vision Pipeline**
   - Reads scanned student drawings.
   - Applies image preprocessing.
   - Uses skeletonization to preserve the Gestalt structure.
   - Extracts structural, spatial, and pressure-related features.
   - Compares student drawings with reference templates.

2. **Machine Learning Pipeline**
   - Processes the Excel clinical classification file.
   - Converts categorical and multi-label observations into numerical features.
   - Trains baseline ML models to predict the final pathological classification.

---

## Dataset

The dataset contains:

- 89 student cases
- Scanned student drawing images
- Excel file with manual classifications and final diagnosis
- 9 original Bender reference template figures

Target column:

```text
التصنيفات المرضية
```

Target distribution after cleaning:

| Class | Count |
|---|---:|
| سوي | 84 |
| العصاب | 3 |
| التخلف العقلي | 2 |

The dataset is highly imbalanced because most samples belong to the normal class.

---

## Project Structure

```text
bender-gestalt-project/
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── excel/
│   │   └── templates/
│   │
│   └── processed/
│       ├── excel_features_clean.csv
│       ├── image_mapping.csv
│       ├── image_features.csv
│       ├── template_features.csv
│       ├── template_deviation_features.csv
│       ├── excel_ml_results.csv
│       └── processed_images/
│           ├── gray/
│           ├── binary/
│           └── skeleton/
│
├── notebooks/
│   ├── 01_excel_feature_engineering.ipynb
│   ├── 02_excel_ml_baseline.ipynb
│   ├── 03_image_preprocessing.ipynb
│   ├── 04_cv_feature_extraction.ipynb
│   ├── 04b_template_deviation_features.ipynb
│   └── 05_image_features_ml.ipynb
│
├── src/
│
├── report/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the notebook kernel:

```bash
python -m ipykernel install --user --name bender_project --display-name "Python (Bender Project)"
```

---

## Requirements

```text
pandas
numpy
openpyxl
scikit-learn
matplotlib
opencv-python
scikit-image
ipykernel
imbalanced-learn
```

---

## Workflow Summary

```text
Excel File
    ↓
Feature Engineering
    ↓
ML Baseline

Scanned Images
    ↓
Image Preprocessing
    ↓
Skeletonization
    ↓
Image Feature Extraction
    ↓
Image-only ML Experiment

Reference Templates
    ↓
Template Preprocessing
    ↓
Template Feature Extraction
    ↓
Template Deviation Features
```

---

## 1. Excel Feature Engineering

Notebook:

```text
notebooks/01_excel_feature_engineering.ipynb
```

The Excel file contained manual clinical classifications and final diagnosis.

### Steps Performed

- Loaded the original `.xlsx` file.
- Removed irrelevant columns such as:
  - Student names
  - Direct case identifiers
  - General comment text
- Cleaned text labels and target values.
- Normalized inconsistent label variants.
- Split multi-value clinical columns into binary features.
- Applied one-hot encoding to categorical columns.
- Converted boolean values into `0/1`.
- Removed constant columns.
- Saved the final processed dataset.

### Multi-value Columns

The following columns contained more than one clinical observation inside the same cell:

```text
التغيرات في الجشطالت
ملاحظات دالة بشكل عام
```

Example:

```text
صعوبات الإغلاق, صعوبات التقاطع, التغير في الانحناءات
```

was converted into separate binary columns:

```text
gestalt_صعوبات الإغلاق
gestalt_صعوبات التقاطع
gestalt_التغير في الانحناءات
```

### Output

```text
data/processed/excel_features_clean.csv
```

Shape:

```text
(89, 34)
```

---

## 2. Excel Machine Learning Baseline

Notebook:

```text
notebooks/02_excel_ml_baseline.ipynb
```

The first machine learning experiment was performed using the processed Excel features.

### Models Tested

- Logistic Regression
- Random Forest
- SVM with RBF kernel

### Evaluation Strategy

Since the dataset contains only 89 cases, a normal train/test split would be unstable.

Therefore, we used:

- Stratified K-Fold Cross Validation
- Accuracy
- Macro F1-score
- Confusion Matrix

Macro F1-score was important because the dataset is highly imbalanced.

### Results

| Model | Accuracy Mean | Accuracy Std | Macro F1 Mean | Macro F1 Std |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.943791 | 0.035161 | 0.554897 | 0.229786 |
| Random Forest | 0.943791 | 0.001307 | 0.485541 | 0.000346 |
| SVM | 0.954902 | 0.022585 | 0.588398 | 0.205801 |

Best model:

```text
SVM
```

### SVM Confusion Matrix

| Actual \ Predicted | التخلف العقلي | العصاب | سوي |
|---|---:|---:|---:|
| التخلف العقلي | 0 | 0 | 2 |
| العصاب | 0 | 1 | 2 |
| سوي | 0 | 0 | 84 |

### Interpretation

- The model correctly predicted all normal cases.
- It detected only one `العصاب` case.
- It failed to detect `التخلف العقلي` cases.
- The high accuracy is misleading because the dataset is dominated by the normal class.
- Macro F1-score gives a more realistic view of performance.

---

## 3. Image Mapping

The scanned image filenames contained page numbers.

These page numbers were extracted and used as `case_id` values to match each image with the corresponding Excel row.

Output:

```text
data/processed/image_mapping.csv
```

This step was important to make sure that each student image was correctly linked with its classification data.

---

## 4. Image Preprocessing

Notebook:

```text
notebooks/03_image_preprocessing.ipynb
```

The preprocessing pipeline was applied to the scanned student images.

### Pipeline

```text
Raw Image
    ↓
Grayscale
    ↓
Median Blur
    ↓
CLAHE
    ↓
Adaptive Thresholding
    ↓
Binary Image
    ↓
Skeletonization
```

### Why Each Step Was Used

| Step | Purpose |
|---|---|
| Grayscale | Preserves pencil pressure and stroke intensity |
| Median Blur | Reduces paper grain while preserving faint strokes |
| CLAHE | Enhances faint pencil strokes and local contrast |
| Adaptive Thresholding | Separates drawing strokes from background |
| Binary Image | Represents stroke presence |
| Skeletonization | Simplifies strokes while preserving Gestalt structure |
| No Rotation | Preserves clinically meaningful orientation |

The images were not rotated because orientation may be meaningful in Bender Gestalt scoring.

Both portrait and landscape images were treated as valid inputs.

### Outputs

```text
data/processed/processed_images/gray/
data/processed/processed_images/binary/
data/processed/processed_images/skeleton/
```

---

## 5. Image Feature Extraction

Notebook:

```text
notebooks/04_cv_feature_extraction.ipynb
```

After preprocessing, numerical features were extracted from the student images.

Output:

```text
data/processed/image_features.csv
```

Shape:

```text
(89, 31)
```

### Extracted Feature Groups

| Feature Group | Examples | Purpose |
|---|---|---|
| Page orientation | image_width, image_height, is_landscape | Handle portrait and landscape scans |
| Ink amount | ink_pixels, ink_density | Measure how much drawing exists on the page |
| Stroke pressure | stroke_darkness_mean, stroke_darkness_std | Capture pencil pressure and stroke intensity |
| Skeleton structure | skeleton_length, skeleton_density | Measure simplified structural length |
| Components and contours | connected_components_count, contour_count | Measure fragmentation and disconnected parts |
| Topology | endpoints_count, intersections_count | Measure line endings and crossings |
| Ratios | endpoints_per_skeleton_length, intersections_per_skeleton_length | Normalize topology by drawing length |
| Spatial layout | bbox_area_ratio, bbox_aspect_ratio, bbox_center_x_norm | Measure drawing location and spread |
| Error proxies | perseveration_score, distortion_proxy_score | Approximate clinical drawing errors |

### Image Feature Columns

```text
case_id
image_width
image_height
is_landscape
ink_pixels
ink_density
stroke_darkness_mean
stroke_darkness_std
skeleton_length
skeleton_density
stroke_thickness_proxy
connected_components_count
small_components_count
largest_component_area
largest_component_area_ratio
mean_component_area
contour_count
endpoints_count
intersections_count
endpoints_per_skeleton_length
intersections_per_skeleton_length
components_per_ink_area
bbox_width
bbox_height
bbox_area_ratio
bbox_aspect_ratio
bbox_center_x_norm
bbox_center_y_norm
drawing_center_distance_norm
perseveration_score
distortion_proxy_score
```

---

## 6. Perseveration and Distortion Scores

Two proxy scores were created to approximate important Bender Gestalt error types.

### Perseveration Score

Perseveration refers to inappropriate repetition or persistence in drawing.

It was approximated using:

- Stroke thickness
- Ink density
- Stroke darkness

Formula:

```text
perseveration_score = mean z-score of:
stroke_thickness_proxy, ink_density, stroke_darkness_mean
```

### Distortion Proxy Score

Distortion refers to structural deformation or abnormal drawing organization.

It was approximated using:

- Endpoints ratio
- Intersections ratio
- Connected components
- Bounding box area ratio

Formula:

```text
distortion_proxy_score = mean z-score of:
endpoints ratio, intersections ratio, connected components count, bbox area ratio
```

---

## 7. Reference Template Analysis

Notebook:

```text
notebooks/04b_template_deviation_features.ipynb
```

The project included 9 original Bender reference template figures.

These templates were processed using the same image preprocessing pipeline:

```text
Template Image
    ↓
Grayscale
    ↓
Binary
    ↓
Skeleton
    ↓
Template Features
```

### Template Feature Output

```text
data/processed/template_features.csv
```

Shape:

```text
(9, 13)
```

Template feature columns:

```text
template_id
ink_pixels
ink_density
skeleton_length
skeleton_density
endpoints_count
intersections_count
connected_components_count
contour_count
bbox_width
bbox_height
bbox_area_ratio
bbox_aspect_ratio
```

---

## 8. Template Deviation Features

Each student page was compared against an aggregated reference summary of the 9 templates.

Output:

```text
data/processed/template_deviation_features.csv
```

Shape:

```text
(89, 9)
```

Columns:

```text
case_id
skeleton_length_deviation
endpoints_deviation
intersections_deviation
components_deviation
contours_deviation
ink_density_deviation
skeleton_density_deviation
template_deviation_score
```

### Deviation Formula

```text
feature_deviation = |student_feature - reference_feature| / (reference_feature + ε)
```

where `ε` is a small value used to avoid division by zero.

### Notes

The current template comparison is page-level.

This means each full student page was compared against an aggregated summary of the 9 reference figures.

A more advanced version would:

- Segment each student page into 9 individual figures.
- Compare each student figure with its corresponding reference template.
- Use shape similarity metrics such as Hausdorff distance or contour matching.

---

## 9. Image-only Machine Learning

An image-feature-only ML experiment was performed using the extracted image features.

The model mostly predicted the majority class:

```text
سوي
```

This happened because the dataset contains only 5 abnormal cases out of 89.

This result shows that:

- Image features are useful for structural representation.
- The dataset is too small for reliable abnormal class prediction.
- Severe class imbalance strongly affects image-only classification.

---

## 10. Key Mathematical Measures

```text
image_area = image_width × image_height
```

```text
ink_density = ink_pixels / image_area
```

```text
skeleton_density = skeleton_pixels / image_area
```

```text
stroke_thickness_proxy = ink_pixels / (skeleton_length + ε)
```

```text
endpoint_ratio = endpoints_count / (skeleton_length + ε)
```

```text
intersection_ratio = intersections_count / (skeleton_length + ε)
```

```text
feature_deviation = |student_feature - reference_feature| / (reference_feature + ε)
```

`ε` is a small value used to avoid division by zero.


---

## 12. Limitations

- The dataset contains only 89 cases.
- The dataset is severely imbalanced.
- Only 5 abnormal cases are available.
- The image-only model mostly predicted the majority class.
- Template comparison was page-level, not figure-level.
- Individual figure segmentation was not implemented.
- No clinical validation was performed.
- More balanced data is required for reliable abnormal class prediction.

---

## 13. Future Work

- Segment each student page into the 9 individual Bender figures.
- Compare each student drawing with its corresponding reference template.
- Use shape similarity metrics such as:
  - Hausdorff distance
  - Chamfer distance
  - Contour matching
  - Graph-based skeleton matching
- Collect more abnormal samples.
- Use object detection or transfer learning after proper annotation.
- Add explainability methods.
- Validate the system with neuropsychology experts.

---

## Notes

Raw student data and scanned images are not uploaded to GitHub because they may contain private student information.

The repository contains code, notebooks, and generated processed outputs only when appropriate.

---

##  Connect with me
Email: moaazmohamedsoliman1@gmail.com
