# ML Lab 7: Gaussian Naive Bayes – Glass Type Classification

## Overview
This practical implements a **Gaussian Naive Bayes (GaussianNB) classifier** to predict the type of glass using its chemical composition. The project uses the Glass Identification dataset and evaluates the model's performance using classification metrics.

## Objectives
- Load and explore the Glass Identification dataset.
- Handle duplicate records and encode target labels.
- Split the dataset into training and testing sets.
- Train a Gaussian Naive Bayes classification model.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Experiment with the `var_smoothing` hyperparameter.

## Dataset
**File:** `glass.csv`

The dataset contains measurements of the chemical composition of glass samples.

Features used:
- Refractive Index (`RI`)
- Sodium (`Na`)
- Magnesium (`Mg`)
- Aluminium (`Al`)
- Silicon (`Si`)
- Potassium (`K`)
- Calcium (`Ca`)
- Barium (`Ba`)
- Iron (`Fe`)

**Target:** `Type` — the glass category.

After removing the duplicate record, the dataset contains **213 rows** and 10 columns.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (if used for visualizations)
- Jupyter Notebook

## Procedure
1. Import the required Python libraries.
2. Load `glass.csv` into a Pandas DataFrame.
3. Inspect the dataset using `info()`, `describe()`, and `shape`.
4. Check for missing values and duplicate records.
5. Remove duplicate records.
6. Encode the glass type labels using `LabelEncoder`.
7. Separate the chemical features (`X`) and target (`y`).
8. Split the dataset into training and testing sets using an 80:20 ratio.
9. Train the Gaussian Naive Bayes classifier.
10. Evaluate the predictions and experiment with `var_smoothing`.

## Model Details
**Algorithm:** Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features are conditionally independent given the class and that continuous features follow a Gaussian distribution within each class.

The dataset is split using:
- Training set: 80%
- Testing set: 20%
- Random state: `42`
- Stratification: Enabled

## Results
The reported model performance on the test set is:

| Metric | Score |
|---|---:|
| Accuracy | 55.81% |
| Weighted Precision | 55.13% |
| Weighted Recall | 55.81% |
| Weighted F1-score | 51.91% |

The confusion matrix and classification report help identify which glass types the model predicts correctly and which classes are frequently confused.

The `var_smoothing` parameter was also tested. The reported weighted F1-score showed little improvement during tuning.

## How to Run
1. Clone or download this repository.
2. Make sure Python is installed.
3. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. Place `glass.csv` in the appropriate working directory.
5. Open the practical notebook:

   ```bash
   jupyter notebook
   ```

6. Open `ML_Prac_7.ipynb` and run the cells in sequence.

## Files
- `ML_Prac_7.ipynb` — notebook containing the practical implementation.
- `glass.csv` — dataset used for classification.

## Conclusion
The practical demonstrates how Gaussian Naive Bayes can classify glass types based on chemical composition. The model achieved **55.81% test accuracy** on the cleaned dataset. The results also demonstrate the importance of evaluating individual class performance, particularly when some classes have fewer samples.

## Author
**Krishna Gupta**

Roll No.: 10  
Section: A-A1
