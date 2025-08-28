# UNSW-NB15-Project
Final year project MSc Data science
# Intrusion Detection on UNSW‑NB15 (Classical ML vs. LSTM/GRU)

This repository contains a complete workflow for building and evaluating intrusion detection models on the **UNSW‑NB15** dataset. It compares **classical machine‑learning baselines** (Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM, MLP) with **deep learning** models (**LSTM** and **GRU**) across **binary** (normal vs. attack) and **multiclass** (attack categories) tasks.

> **Highlights from the latest runs**
>
> - **Binary classification**: Recurrent models led the leaderboard, with **GRU reaching perfect scores (Accuracy/Precision/Recall/ROC‑AUC ≈ 1.000)** on the test split used in the notebook.
> - **Multiclass classification**: **LSTM achieved ~86.6% accuracy** and the best F1 among tested models on the same split.
>
> These figures depend on the random split, preprocessing, and exact hyperparameters in the notebook; see “Reproducibility” below.

---

## Project Structure

```
.
├─ report.ipynb           # Main Colab-friendly notebook that runs end-to-end
└─ README.md              # This file
```

The notebook is organized in clearly labeled sections:
- **Configure Kaggle & Download Dataset**
- **Read & Concatenate Train/Test Data**
- **Preprocess the Dataset**
- **Exploratory Data Analysis** (incl. label distribution and protocol by attack category)
- **Split the Dataset** (80/20 train–test with `random_state=17`)
- **Model Development**
  - **Machine Learning Models** (LogReg, KNN, GNB, MLP)
  - **Deep Learning Models** (LSTM, GRU)
- **Evaluation & Plots** (metrics, confusion matrices/ROC where applicable)

---

## Data

The notebook expects the **UNSW‑NB15** train/test files in Parquet format (downloaded via Kaggle CLI):
```
/content/unsw_nb15/UNSW_NB15_training-set.parquet
/content/unsw_nb15/UNSW_NB15_testing-set.parquet
```

### How the data is used
- Both **binary** (`label`) and **multiclass** (`attack_cat`) targets are supported.
- Categorical/text columns are **label‑encoded**.
- All features are **scaled** with **`MinMaxScaler`** prior to model training.
- An **80/20** train/test split is applied with `random_state=17`.

> The **effective number of features** depends on which target you select and the columns retained after preprocessing. The notebook constructs:
> ```python
> X = df_clean.drop(columns=['label'])   # for binary
> y = df_clean['label']
> 
> X = df_clean.drop(columns=['attack_cat'])  # for multiclass
> y = df_clean['attack_cat']
> ```

---

## Environment & Requirements

The notebook is designed for **Google Colab**, but you can run it locally.

**Python 3.9+** recommended. Core packages:
- `pandas`, `numpy`, `scikit-learn`
- `matplotlib`
- `tensorflow` (for LSTM/GRU)
- `nbformat` (if inspecting the notebook programmatically)
- (Optional) Kaggle CLI if downloading inside the notebook

Install locally:
```bash
pip install pandas numpy scikit-learn matplotlib tensorflow kaggle
```

---

## Getting the Dataset (Colab/Kaggle CLI)

In Colab you’ll be prompted to upload your `kaggle.json` and run Kaggle CLI commands in the **“Configure Kaggle & Download Dataset”** cell. This will download and place the Parquet files under `/content/unsw_nb15/` as expected by the notebook.

If running **locally**, either:
1. Use Kaggle CLI to download the dataset to a local folder and update the paths in the notebook, or
2. Place the Parquet files at the same paths used in the notebook and run as‑is.

---

## Preprocessing

- **Label encoding** of all `object`/categorical columns.
- **Train/test split**: `train_test_split(..., test_size=0.2, random_state=17)`.
- **Scaling**: `MinMaxScaler` fitted on `X_train`, applied to `X_test`.
- **Reshaping for RNNs**: features reshaped to `(samples, 1, features)` to feed LSTM/GRU.

---

## Models

### Classical ML (binary & multiclass sections)
- **Logistic Regression** (`max_iter=1000`, `random_state=17`)
- **K‑Nearest Neighbors (KNN)`**
- **Gaussian Naive Bayes (GNB)`**
- **Decision Tree`**
- **Support Vector Classifier (SVC)`**
- **Multi‑Layer Perceptron (MLP)`**

> Models are trained with default or common‑sense hyperparameters, and the notebook records **training time**, **test time**, and standard **metrics**.

### Deep Learning

Two recurrent architectures are implemented using Keras/TensorFlow:

#### LSTM (typical config used in the notebook)
- **Layers**: LSTM(64) → Dropout(0.2) → LSTM(32) → Dense output
- **Activation**:
  - **Binary**: final **`sigmoid`**
  - **Multiclass**: final **`softmax`** with one‑hot targets
- **Optimizer**: `Adam(learning_rate=1e-3)`
- **Epochs**: `50`
- **Batch size**: `64`
- **Validation split**: `0.2`

#### GRU (typical config used in the notebook)
- **Layers**: GRU(64) → Dropout(0.2) → GRU(32) → Dense output
- **Activation**: `sigmoid` (binary) or `softmax` (multiclass)
- **Optimizer**: `Adam(learning_rate=1e-3)`
- **Epochs**: `50`
- **Batch size**: `64`
- **Validation split**: `0.2`

> The notebook uses **ModelCheckpoint** to save the best weights during training.

---

## Evaluation

For **binary** tasks, the notebook reports:
- **Accuracy**, **Precision**, **Recall**, **F1**
- (Where implemented) **ROC‑AUC**, ROC curves, confusion matrix

For **multiclass**, it uses:
- **Accuracy**
- Class‑wise and macro **Precision/Recall/F1**
- Confusion matrix

> **Binary highlight**: GRU achieved ≈ **1.000** across Accuracy/Precision/Recall/ROC‑AUC on the provided split.
>
> **Multiclass highlight**: LSTM achieved ≈ **0.866** accuracy with the best F1 among tested models.

Your numbers may vary depending on randomness, class balancing, and exact preprocessing choices.

---

## How to Run

### Option A — Colab (recommended)
1. Open **`report.ipynb`** in Google Colab.
2. Run the **Kaggle** setup cell and download the dataset (upload your `kaggle.json` when prompted).
3. Execute cells from top to bottom. Choose **binary** or **multiclass** section as needed.
4. Review printed metrics and generated plots (label distribution, confusion matrices, etc.).

### Option B — Local
1. Install requirements (see above).
2. Download the dataset to a known location.
3. Update paths in the notebook to point to your local files.
4. Run the notebook (e.g., in VS Code, Jupyter Lab, or `jupyter notebook`).

---

## Reproducibility

- The **train/test split** is fixed with `random_state=17`.
- For full reproducibility of deep learning results, set framework seeds (TensorFlow, NumPy, Python) and consider disabling non‑deterministic cuDNN ops.
- Results can differ by **hardware**, **driver**, and **library versions**.

---

## Notes & Tips

- **Feature count**: Will differ based on the columns retained after preprocessing. You can print `X.shape` after constructing `X` to confirm (e.g., `print(X.shape[1])`).
- **Binary vs. Multiclass**: The notebook contains separate cells that redefine `X`/`y` for each target (`label` vs `attack_cat`). Run only the relevant block at a time for clarity.
- **Scaling**: `MinMaxScaler` is applied consistently before both classical and deep models.

---

## Citation

If you use this work, please cite the dataset and any papers/tools you rely on. Example dataset citation (adapt to the specific Kaggle source you use) and any relevant IDS papers you referenced in your report.

---

## License

This project is for academic use. Add a lice
