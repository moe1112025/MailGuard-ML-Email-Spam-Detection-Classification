# MailGuard ML — Email Spam Detection & Classification

MailGuard ML is a standalone text-classification system that compares multiple classical machine-learning pipelines and persists the selected classifier as a single inference artifact.

## Capabilities

- CSV schema validation
- Duplicate removal
- Stratified train/test split
- TF-IDF word and bigram representation
- Naive Bayes, Logistic Regression, Linear SVM, Random Forest and Decision Tree baselines
- Stratified cross-validation with GridSearchCV
- Accuracy, precision, recall, F1 and ROC-AUC evaluation
- JSON model report
- Streamlit inference console
- Local demo dataset included

## Run

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Windows shortcut:

```bat
run.bat
```

## Dataset

The project includes `data/email.csv` so the application can be run immediately. It is a small demonstration corpus and should not be treated as a production benchmark.

For larger evaluation, the UCI SMS Spam Collection is a suitable public reference dataset:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Architecture

```text
CSV -> validation -> cleaning -> stratified split -> TF-IDF -> model search -> evaluation -> best pipeline -> artifact -> inference UI
```

Keeping TF-IDF inside the model pipeline prevents feature fitting from leaking across cross-validation folds.

## Technology

Python: https://www.python.org/
scikit-learn: https://scikit-learn.org/stable/
Streamlit: https://docs.streamlit.io/
pandas: https://pandas.pydata.org/docs/

## Model Interpretation

A spam prediction is a statistical classification signal. Production mail security should combine classifier output with sender reputation, authentication signals, sandboxing and policy controls.

## License

Project source: MIT. Dataset terms remain applicable.
