# XAI-PGM-Counterfactuals

This repository contains the code from my MSc dissertation work on explainable AI for medical-image classification.

The starting point is simple: a CNN can classify a chest X-ray as Normal or Pneumonia, but the prediction alone is not enough. I wanted to test whether the internal CNN features could be transformed into a more structured explanation, using probabilistic reasoning instead of only heatmaps.

The project combines three ideas:

1. CNN feature extraction
2. Region-based prototype similarity
3. Bayesian Network counterfactual analysis

The goal is not to claim clinical diagnosis. The goal is to explore whether CNN feature regions can be connected with a probabilistic model so that we can ask counterfactual questions such as:

```text
What happens to the predicted class probability if a specific image region changes from medium similarity to high similarity?
```

---

## Problem I worked on

Deep-learning models can perform well on image-classification tasks, but their reasoning is not directly visible. Standard XAI methods such as Grad-CAM, LIME, and SHAP produce useful visual explanations, but they do not give a structured probabilistic model of how regions relate to each other.

My idea was to use the CNN as the feature extractor and then build a Bayesian Network over region-level feature states.

In plain terms:

```text
CNN tells us what it sees.
The Bayesian Network is used to examine how region-level evidence is probabilistically connected.
Counterfactuals test how changing one region state affects the final probability.
```

---

## Dataset

- Dataset: Chest X-Ray Images (Pneumonia), Kaggle
- Task: Binary classification, Normal vs Pneumonia
- Image size used in the model: 150 x 150
- Region setup: 3 x 3 image grid

Important note: the 9 regions are positional grid regions. They are not manually labelled anatomical lung regions. I made this choice because the method had to remain automatic and not depend on manual medical annotation.

| Region | Position |
|---|---|
| region_0 | upper-left |
| region_1 | upper-center |
| region_2 | upper-right |
| region_3 | middle-left |
| region_4 | center |
| region_5 | middle-right |
| region_6 | lower-left |
| region_7 | lower-center |
| region_8 | lower-right |

---

## Method

### 1. Train the CNN

I trained a CNN for the Normal vs Pneumonia classification task.

The architecture is intentionally simple:

```text
Input image
-> Conv2D + MaxPooling
-> Conv2D + MaxPooling
-> Conv2D + MaxPooling
-> Flatten
-> Dense
-> Dropout
-> Sigmoid output
```

The CNN is the classifier. The Bayesian Network is not used to replace the CNN. It is used later for explanation.

### 2. Extract features from the last convolutional layer

I used the last convolutional layer because it contains more abstract visual patterns than the early layers, while still preserving spatial structure.

This matters because the explanation is region-based. If spatial information is completely lost, the Bayesian Network would have no meaningful regional evidence to model.

### 3. Split every image into 9 regions

Each image is divided into a 3 x 3 grid. For each region, CNN features are extracted and converted into a compact representation.

### 4. Build regional prototypes

For every region:

```text
CNN features
-> global average pooling
-> PCA dimensionality reduction
-> K-Means prototype
-> cosine similarity to prototype
```

The prototype is not a medical concept. It is a learned feature reference point for a specific region.

### 5. Discretize similarity scores

The cosine similarity values are converted into three states:

```text
low, medium, high
```

These states become the variables used by the Bayesian Network.

### 6. Learn the Bayesian Network

The Bayesian Network is learned from the region-state data.

- Nodes: region_0 to region_8, plus label
- Structure learning: Hill Climb Search with K2 Score
- Parameter estimation: Maximum Likelihood Estimation

The BN is used as a structured probabilistic explanation layer over the CNN-derived regional evidence.

### 7. Generate counterfactual explanations

For each region, I change its state and observe how the predicted class probability changes.

Example:

```text
region_7: medium -> high
original probability: 52.53%
new probability: 84.00%
change: +31.34%
```

This gives a rule-style explanation:

```text
Changing region_7 from medium to high similarity increases the probability of the Pneumonia class in this experiment.
```

---

## Results from the experiment

### Counterfactual region changes

| Changed region | Change | Original probability | New probability | Difference |
|---|---:|---:|---:|---:|
| region_7 | Medium -> High | 52.53% | 84.00% | +31.34% |
| region_7 | Medium -> Low | 52.53% | 26.00% | -26.84% |
| region_4 | Medium -> Low | 52.53% | 29.00% | -23.16% |
| region_0 | Low -> Medium | 52.53% | 68.00% | +15.50% |
| region_4 | Medium -> High | 52.53% | 66.00% | +13.72% |

In this run, region_7 had the strongest counterfactual effect. I would not interpret this medically by itself. The safer interpretation is that the lower-center grid region had the strongest influence inside this learned feature-probability structure.

---

## Comparison with other XAI methods

I compared the region ranking from the Bayesian Network counterfactuals with Grad-CAM, LIME, and SHAP.

| Method | Spearman rho | p-value | Weighted ranking score | p-value |
|---|---:|---:|---:|---:|
| LIME | 0.133 | 0.733 | 0.8973 | 0.001 |
| SHAP | 0.300 | 0.4328 | 0.9568 | 0.0001 |
| Grad-CAM | 0.7667 | 0.0159 | 0.9899 | 0.0001 |

Grad-CAM aligned most strongly with the PGM-based counterfactual ranking in this experiment.

I treat this as supporting evidence, not final proof. The sample size, discretization method, and grid-based regions all affect the result.

---

## Visual examples

### Bayesian Network

![Bayesian Network](images/BN.png)

### Training and validation curves

![Training and Validation](images/Training_Validation.png)

### 3 x 3 image grid

![Divided X-ray](images/Xray_divided.png)

### LIME

![LIME Explanation](images/LIME.png)

### SHAP

![SHAP Explanation](images/SHAP.png)

### Grad-CAM

![Grad-CAM Explanation](images/GradCAM.png)

---

## Repository structure

```text
.
├── Evangelos_Kitsoulis_Dissertation's_Python_Code.py
├── images/
│   ├── BN.png
│   ├── GradCAM.png
│   ├── LIME.png
│   ├── SHAP.png
│   ├── Training_Validation.png
│   └── Xray_divided.png
├── README.md
├── LICENCE
└── requirements.txt
```

---

## How to run

Clone the repository:

```bash
git clone https://github.com/ekitsoulis/XAI-PGM-Counterfactuals.git
cd XAI-PGM-Counterfactuals
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle Chest X-Ray Images dataset and update the local path in the script:

```python
data_dir = "path/to/chest_xray"
```

The original code was developed locally on Windows, so the path must be changed before running it elsewhere.

---

## What I would improve next

This project works as a research prototype, but it is not yet a clean production-style package.

The most important improvements would be:

1. Split the code into modules instead of one long exported script.
2. Move dataset paths and experiment settings into a config file.
3. Add a small demo notebook that runs on a limited sample.
4. Save all outputs into a structured `results/` folder.
5. Add tests for image splitting, feature extraction, similarity calculation, and counterfactual generation.
6. Run the BN stage on a larger sample if memory allows.

---

## Limitations

The main limitations are important:

- The 3 x 3 grid is automatic but crude.
- The regions are not medically annotated.
- The Bayesian Network shows learned statistical dependencies, not clinically validated causality.
- The counterfactuals depend on discretization into low, medium, and high states.
- The BN stage was limited by available memory.
- The work should be read as an explainability experiment, not as a medical tool.

This is the central trade-off of the project: I avoided manual medical labels and expert annotation, but the price is that the explanations are spatial and probabilistic rather than anatomical and clinically verified.