# SMS & Email Spam Classifier: SVM from Scratch

An end-to-end Machine Learning pipeline that classifies text messages as either **Ham (Legitimate)** or **Spam**. This project features a custom implementation of a **Support Vector Machine (SVM)** classifier built completely from scratch using Python and linear algebra, benchmarked directly against industry-standard libraries.

---

## Dataset Information
* **Dataset Used:** SMS Spam Collection Dataset (`spam.csv`).
* **Task:** Binary Text Classification.
* **Target Classes:** `Ham` (0) and `Spam` (1).
* **Features:** Raw text messages containing colloquial expressions, links, and abbreviations.
<img width="709" height="552" alt="image" src="https://github.com/user-attachments/assets/203ba3ef-6726-4ecd-90d5-40e20abc27b0" />

---

##  Pipeline & Project Workflow
The project extracts raw textual datasets and processes them through a mathematical optimization matrix:
### Step-by-Step Overview:
1. **Data Ingestion & Cleaning:** Loads text data using `pandas`, handles character encoding, drops null attributes, and tokenizes strings by removing numeric characters, symbols, and punctuation marks.
2. **Feature Extraction (TF-IDF):** Transforms unstructured text into numerical vector arrays using Term Frequency-Inverse Document Frequency (TF-IDF). This scores words based on how unique they are to specific messages relative to the wider text corpus.
3. **Custom SVM Architecture:** Implements a binary Support Vector Machine linear classifier from scratch. It uses standard matrix computations to calculate separating hyperplanes that maximize margins between data classes.
4. **Scikit-Learn Benchmarking:** Employs Scikit-Learn's optimized `SVC` module to act as a control variant baseline for performance checking.
5. **Evaluation Matrix:** Evaluates mathematical validation metrics (Accuracy, Precision, Recall, or F1 scores) across the trained systems to see how well the scratch model keeps up with the optimized library.
6. **Inference Test:** Runs arbitrary custom input phrases through the pipeline to determine instantly whether a message is an unwanted advertisement or an authentic text.

---

## Metrics Visualizations
<img width="674" height="699" alt="image" src="https://github.com/user-attachments/assets/da5c1290-a10d-4d39-8442-4595003757ce" />


### Model Performance Matrix Comparison
<img width="732" height="456" alt="image" src="https://github.com/user-attachments/assets/6b3dc6f9-debd-4d1c-83f2-f776d12f5a09" />

### Single-Sample Inference Output

---
## Tech Stack & Prerequisites
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Text Engineering & Math Vectorization:** Scikit-Learn (`TfidfVectorizer`)
* **Evaluation Algorithms:** Scikit-Learn Metrics

<img width="700" height="87" alt="image" src="https://github.com/user-attachments/assets/813eeb10-9638-4d54-90a9-be17422171d7" />

---
##  Quick User Guide

Since this project is fully self-contained within a Jupyter Notebook, running it requires minimal configuration:

### Running via Web Browser (Google Colab - Recommended)
1. Navigate to the `HAM_VS_SMAM_IR.ipynb` notebook file uploaded within this repository.
2. Click the **Open in Colab** badge (if available) or open it inside your preferred cloud notebook provider.
3. Upload your `spam.csv` data asset file directly into your runtime storage environment.
4. Execute the cells sequentially from top to bottom.

### Running Locally
Ensure your local Python libraries are satisfied before compiling:
```bash
pip install pandas numpy scikit-learn
