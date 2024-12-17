# StressDeduction-Text_Analysis

## 📜 Overview
This project leverages **Natural Language Processing (NLP)** techniques and **Machine Learning** to analyze text data and classify content as either **Stress** or **Not Stress**. The goal is to preprocess textual data, build a predictive model, and evaluate its ability to detect stress levels accurately.

---

## 🚀 Features

- **Text Preprocessing**:
  - Converts all text to lowercase.
  - Removes punctuation, special characters, and numbers.
  - Removes stopwords (e.g., "the", "is") to focus on meaningful words.
  - Applies stemming to reduce words to their root forms (e.g., "running" → "run").

- **Machine Learning**:
  - A classifier is trained to predict **Stress (1)** or **Not Stress (0)** based on the processed text.

- **Visualization**:
  - Provides insights and visualizations of data distributions through graphs and plots.

- **Simple and Extendable**:
  - Modular code allows easy improvements, such as integrating advanced models or datasets.

---

## 🧰 Technologies Used

The following tools and libraries are used in this project:

- **Python**: Core programming language.
- **Jupyter Notebook**: Interactive environment for running code and visualizing outputs.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **NLTK (Natural Language Toolkit)**: Text preprocessing (stopword removal, stemming, etc.).
- **Scikit-learn**: Building and evaluating machine learning models.
- **Matplotlib & Seaborn**: Data visualization and plotting.

---

## 📊 Dataset

The dataset file is named `stress.csv` and contains:

| Column     | Description                                 |
|------------|---------------------------------------------|
| **Text**   | Raw text sentences or paragraphs.           |
| **Label**  | Target column with binary values:           |
|            | - `1` → Stress                              |
|            | - `0` → Not Stress                          |
| **Confidence** _(optional)_ | Indicates labeling confidence level. |

**Example rows from the dataset**:

| Text                                  | Label | Confidence |
|---------------------------------------|-------|------------|
| "I'm very worried about my exams."    | 1     | 0.95       |
| "Feeling great and relaxed today."    | 0     | 0.90       |

---

## ⚙️ Installation

Follow these steps to set up the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/stress-detection.git
   cd stress-detection
   ```

2. **Install the required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

---

## 🧑‍💻 Usage

1. **Load the Dataset**:
   - Open `main.ipynb` in Jupyter Notebook.
   - Run the initial cells to load and explore the dataset.

2. **Text Preprocessing**:
   - The `clean_text` function processes raw text by:
     - Converting to lowercase.
     - Removing punctuation and stopwords.
     - Applying stemming to simplify words.

3. **Train the Model**:
   - Processed text data is fed into a machine learning classifier to train it to predict stress levels.

4. **Make Predictions**:
   - Use the trained model to predict stress levels for new sentences.

   **Example**:
   ```python
   model.predict(["I'm under a lot of pressure."])
   ```

---

## 📈 Results

- **Stress Classification**:
  - Texts indicating stress (e.g., anxiety, worry) are classified as **Stress**.
  - Texts without stress indicators (e.g., happy, relaxed moods) are classified as **Not Stress**.

- **Performance Metrics**:
  - The model's accuracy, precision, recall, and F1-score are evaluated using a test set.
  - Results are visualized using confusion matrices and performance graphs.

---

## 📊 Sample Output

| Input Sentence                          | Predicted Label |
|-----------------------------------------|-----------------|
| "I am feeling very stressed today."     | Stress          |
| "It is a beautiful day outside."        | Not Stress      |

