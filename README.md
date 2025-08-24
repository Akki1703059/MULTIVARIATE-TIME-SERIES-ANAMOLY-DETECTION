# Anomaly Detection in Industrial Processes

## Project Overview

This project implements an unsupervised machine learning solution for **proactive anomaly detection and diagnosis** in a continuous industrial process, specifically using the Tennessee Eastman Process (TEP) dataset. The goal is to identify deviations from normal operating conditions and pinpoint the likely root causes, thereby enhancing operational safety and efficiency.

---

## Features

* **Automated Data Preprocessing**: Handles raw CSV data, including cleaning inconsistent date/time formats.
* **Unsupervised Anomaly Detection**: Employs the `IsolationForest` algorithm to learn normal system behavior and identify abnormal patterns.
* **Abnormality Scoring**: Generates an `Abnormality_score` (0-100) for each data point, indicating the severity of the anomaly.
* **Root Cause Analysis**: Identifies the **top 7 most contributing features** (sensor readings) for each detected anomaly, providing actionable insights.
* **Configurable Parameters**: Designed with configurable variables for easy adaptation to different datasets or timeframes.
* **CSV Output**: Produces a comprehensive CSV file with original data, anomaly scores, and contributing features.

---

## Technologies Used

* **Python 3.x**: The core programming language.
* **`pandas`**: For efficient data manipulation and analysis.
* **`numpy`**: For numerical operations.
* **`scikit-learn`**: For machine learning algorithms, specifically `IsolationForest` and `MinMaxScaler`.

---

## Setup and Installation

To get this project up and running, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone [your-repository-link]
    cd [your-repository-name]
    ```

2.  **Install Python (if not already installed)**:
    Download and install Python 3.x from [python.org](https://www.python.org/downloads/).

3.  **Install required libraries**:
    Open your terminal or command prompt and run:
    ```bash
    pip install pandas numpy scikit-learn
    ```

4.  **Place the Dataset**:
    Download the `TEP_Train_Test.csv` file and place it in the root directory of your project folder (the same folder where `anomaly_detection.py` is located).

---

## Usage

To run the anomaly detection script:

1.  Open your terminal or command prompt.
2.  Navigate to your project directory.
3.  Execute the Python script:
    ```bash
    python anomaly_detection.py
    ```

### Configuration
You can modify the `anomaly_detection.py` script's `--- USER CONFIGURATION ---` section to adjust parameters like `FILE_NAME`, `TIME_COLUMN_NAME`, `TRAIN_START_DATE`, `TRAIN_END_DATE`, `ANALYSIS_START_DATE`, `ANALYSIS_END_DATE`, and `NUM_TOP_FEATURES` for different datasets or analysis requirements.

---

