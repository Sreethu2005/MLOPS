# Dataset Source Document

* **Dataset name**: Diabetes 130-US hospitals for years 1999-2008 Data Set
* **Source**: UCI Machine Learning Repository
* **Access instructions**: Available for public download at [UCI Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) or via ZIP [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip)
* **Why selected**: Explicitly recommended by the project specification for clinical analytics.
* **Description**: The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.
* **Original size**: ~100,000 instances
* **Original columns**: 50 attributes
* **License/access notes**: Public domain / Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Extraction process**: Downloaded as ZIP, extracted CSV `diabetic_data.csv`.
* **Privacy considerations**: De-identified data. All patient identifiers are safe and hashed.
* **Limitations**: Data is relatively old (1999-2008). Time in hospital is in days (integer) rather than exact timestamps. Missing values are prevalent (represented as `?`).
