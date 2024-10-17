# Insurance Premium Explanation Framework

## Overview

This repository contains the implementation of a framework designed to explain insurance premium calculations without direct access to proprietary algorithms. The project aims to address the need for transparency in the insurance industry by employing advanced machine learning techniques to identify key factors influencing premiums across multiple insurance domains.

## Key Features

1. **XGBoost Models**: Utilizes XGBoost to mimic insurance premium algorithms with high accuracy.
2. **SHAP Analysis**: Implements SHAP (SHapley Additive exPlanations) to identify and explain the most influential factors in premium determination.
3. **Meta-Model**: Includes a meta-model that predicts the effectiveness of SHAP analysis, adding an extra layer of reliability to the framework.
4. **Natural Language Processing**: Integrates NLP techniques to generate stakeholder-specific reports, bridging the gap between complex algorithmic outputs and comprehensible explanations.
5. **Cross-Domain Applicability**: Demonstrates robust performance across various insurance domains.

## Repository Structure

```
.
├── .gitignore
├── Calculations/
│   ├── auto_insurance_premium.py
│   ├── cybersecurity_insurance_premium.py
│   ├── env_liability_insurance_premium.py
│   ├── travel_insurance_premium.py
│   └── utilities.py
├── Experiment 1 - Determine Best Model/
│   ├── 01 - Filter Models - Collect Data.py
│   ├── 02 - Filter Models - Result Analysis.ipynb
│   ├── 03 - Determine Best Model - Collect Data.py
│   ├── 04 - Determine Best Model - Result Analysis.ipynb
│   ├── Determine Best Model.csv
│   └── Filter Models Data.csv
├── Experiment 2 - SHAP Precision ML/
│   ├── 01 - Collect Experiment 3 Data.py
│   ├── 02 - Data Analysis.ipynb
│   ├── 03 - Machine Learning Predictor.ipynb
│   ├── Experiment_3_data.csv
│   └── Experiment_3_filtered_data.csv
├── Experiment 3 - Generate Report/
│   └── 01 - Generate Pipeline.ipynb
└── travel_insurance_premium.py
```

## Key Components

1. **Calculations**: Contains modules for different types of insurance premium calculations.
   - `auto_insurance_premium.py`
   - `cybersecurity_insurance_premium.py`
   - `env_liability_insurance_premium.py`
   - `travel_insurance_premium.py`
   - `utilities.py`: Shared utility functions

2. **Experiment 1 - Determine Best Model**: Scripts and notebooks for model selection and evaluation.
   - Data collection scripts
   - Result analysis notebooks
   - CSV files with experiment results

3. **Experiment 2 - SHAP Precision ML**: Implementation of SHAP analysis and related machine learning tasks.
   - Data collection and analysis scripts
   - Machine learning predictor notebooks
   - Experiment data CSV files

4. **Experiment 3 - Generate Report**: Pipeline for generating reports based on the analysis.
   - `01 - Generate Pipeline.ipynb`: Notebook for report generation

5. `travel_insurance_premium.py`: Standalone script for travel insurance premium calculations.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/insurance-premium-explanation.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Navigate through the experiment folders to run specific scripts or notebooks:
   - Use the numbered scripts/notebooks in each experiment folder to replicate the analysis
   - Refer to individual script/notebook documentation for detailed usage instructions

## Results

Our findings reveal that XGBoost consistently outperforms other machine learning models in mimicking the tested insurance premium algorithms, with high accuracy in identifying influential factors. The framework demonstrates robust performance across various insurance domains, offering valuable insights for consumers, regulators, and insurers alike.

## Limitations and Future Work

- The current implementation uses simulated data, which presents certain limitations.
- Future directions include:
  - Validation with real-world data
  - Expansion to additional insurance domains
  - Exploration of potential cross-industry applications

## Contribution

This work contributes significantly to addressing transparency challenges in the insurance industry, with implications for improved decision-making, regulatory oversight, and ethical considerations in algorithmic pricing.

## License

Apache License 2.02

## Contact

Alberto Primerano
alberto.primerano@gmail.com
