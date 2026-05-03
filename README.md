## Project Timeline & Logistics

**Week 2 (Friday):** Final deadline for group formation (3–4 students).

**Week 6 (Friday):** Finalize the Superstore Sales dataset selection and define the retail sales forecasting task.

**May 8 (Friday):** Finalize Home Assignment Submission.

**April 13 – May 14:** Presentation Submission.

**Infrastructure:** All collaborative work must be hosted in a joint GitHub repository containing a shared notebook, README, cleaned datasets, and individual modeling notebooks.

---

## Project Workflow

### Collaborative Foundation (Group Work)

You must collectively perform Exploratory Data Analysis (EDA) on the Superstore Sales dataset to identify sales trends, weekly and monthly seasonality, volatility patterns, and outliers.

This phase includes cleaning the transactional data, resampling it into regular daily and monthly time-series datasets, applying log transformation for variance stabilization, and defining the project’s forecasting framework: target variables, forecast horizons, and validation strategy (chronological train/test split).

---

### Individual Implementation (Personal Responsibility)

Each member is responsible for their own forecasting notebook.

You must implement:

- **One Baseline:** Simple forecasting benchmarks such as Naïve Forecast or Historical Mean.

- **One Unique Model:** A classical statistical model (SARIMA, ETS) or an advanced machine learning model (XGBoost).

- **Crucial:** No two group members can implement the same model type.

---

## Evaluation & Reflection

- Individually, you will test your model against the baseline using multiple evaluation metrics.

- The group will then reconvene to compare forecasting performance across all models, determine which approach best captures retail sales behavior, and reflect on the practical business relevance of the findings for short-term operational planning and medium-term sales forecasting.

---

## Grading Breakdown (100 Points Total)

- **EDA & GitHub (30 Points):** A shared group score based on the depth of data exploration, preprocessing quality, and GitHub organization.

- **Modelling (40 Points):** An individual score reflecting your ability to implement, optimize, and justify your forecasting model.

- **Presentation (20 Points):** An individual score for your 5-minute Jupyter notebook walkthrough and your ability to answer technical questions.

- **Coding Quality (10 Points):** Evaluated on the cleanliness, modularity, and readability of your Python code.

---

## Submission Requirements

Every student must upload two files to Moodle:

- **The Joint Notebook**
(containing the EDA, preprocessing pipeline, forecasting problem definition, and GitHub link).

- **Your Individual Notebook**
(containing your specific forecasting model, tuning process, evaluation results, and reflection).
