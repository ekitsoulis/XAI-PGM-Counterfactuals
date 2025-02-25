# 🧠 XAI-PGM-Counterfactuals: Explainable AI with Probabilistic Graphical Models

## 📌 Overview
This project presents a **novel Explainable AI (XAI) framework** integrating **Convolutional Neural Networks (CNNs)** and **Probabilistic Graphical Models (PGMs)** to provide **counterfactual explanations** for AI-driven medical diagnoses. The methodology focuses on **interpreting CNN predictions** in a **pneumonia classification task** using **chest X-ray images**, enhancing transparency in AI-driven decision-making.

### **🔹 Key Contributions**
✔ **CNN Feature Extraction:** Identifies **critical regions** in medical images.  
✔ **Prototype-Based Explainability:** Maps CNN **feature maps** to interpretable regions.  
✔ **Bayesian Networks:** Models relationships **between image regions**.  
✔ **Counterfactual Reasoning:** Evaluates **alternative explanations** for AI predictions.  
✔ **Comparison with LIME, SHAP, and Grad-CAM:** Benchmarks performance with existing XAI techniques.

---

## 📂 Dataset
- **Dataset Used:** [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray) 📸  
- **Task:** **Binary classification** (Normal vs. Pneumonia).
- **Feature Representation:** CNN-extracted features from **segmented lung regions**.

---

## 🚀 Methodology
### **1️⃣ CNN Feature Extraction**
- A **Convolutional Neural Network (CNN)** is trained on chest X-ray images.
- Extracted **feature maps** from the **last convolutional layer**.
- Divided images into a **3×3 grid** for **localized interpretability**.
- **Cosine similarity** is used to compare regions with **prototypes**.

### **2️⃣ Prototype-Based Explainability**
- **K-Means Clustering** applied to extracted features for **regional prototypes**.
- **Principal Component Analysis (PCA)** used for feature reduction.
- Similarity between new images and stored prototypes computed using **cosine similarity**.

### **3️⃣ Probabilistic Graphical Model (PGM) Construction**
- **Bayesian Network (BN)** created where **nodes represent image regions**.
- BN structure **learned using Hill Climb Search + K2 Score**.
- Conditional Probability Distributions (CPDs) estimated using **Maximum Likelihood Estimation**.

### **4️⃣ Counterfactual Generation & Analysis**
- **Region-wise feature similarities** in the Bayesian Network are **modified**.
- Evaluated how **changes in individual regions** affect CNN predictions.
- **Spearman’s Rank Correlation (ρ) and Weighted Spearman’s Rank Correlation (WFMρ)** used to **validate counterfactual results**.
- Counterfactuals compared with **LIME, SHAP, and Grad-CAM**.

---

## 📊 Results & Key Findings

### **Counterfactual Explanations for CNN Predictions**
| Changed Region | Change (From → To) | Original Prob | New Prob | Change in Prob |
|---------------|--------------------|--------------|----------|---------------|
| **Region 7**  | Medium → High      | 52.53%       | 84%      | +31.34%       |
| **Region 7**  | Medium → Low       | 52.53%       | 26%      | -26.84%       |
| **Region 4**  | Medium → Low       | 52.53%       | 29%      | -23.16%       |
| **Region 0**  | Low → Medium       | 52.53%       | 68%      | +15.50%       |
| **Region 4**  | Medium → High      | 52.53%       | 66%      | +13.72%       |

### 🔹 **Key Insights**
✔ **Region 7 is the most critical region** for CNN decision-making.  
✔ Changes in specific regions **significantly impact pneumonia probability**.  
✔ **Bayesian Network effectively models** interdependencies between image regions.  
✔ **Counterfactuals align with LIME, SHAP, and Grad-CAM**, validating interpretability.  

---

