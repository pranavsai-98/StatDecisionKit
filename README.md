# (StatDecisionKit) Automated Statistical Testing and Feature Selection

## Introduction
In the rapidly evolving field of data science, the need for efficient and accurate data analysis is paramount. Our Python library, "StatDecisionKit," is designed to streamline and automate key aspects of the data science pipeline, focusing on statistical testing and feature selection.

## Project Objective
Our primary objective is to create a Python library that simplifies the process of statistical testing and feature selection for data scientists, researchers, and analysts. This tool is particularly beneficial for those without extensive statistical backgrounds.

## Features and Functionalities

### Automated Statistical Testing
- **Input Data Handling**: Users provide a DataFrame along with two features of interest.
- **Test Identification**: The library automatically determines the most suitable statistical test for the given data features.
- **Test Execution and Results**: Performs the test and returns the results, aiding in the decision-making process.

### Integrated Feature Selection
- **Comprehensive Analysis**: Users pass the entire DataFrame and specify the dependent variable.
- **Selective Testing**: Conducts statistical tests across all independent variables.
- **Result Optimization**: Identifies and retains only statistically significant variables, thereby enhancing model accuracy.

### Language and Tools
The library is developed in Python, leveraging popular libraries such as Pandas, SciPy, and NumPy.

## Target Audience
This library is ideal for data scientists, researchers, and analysts involved in data processing and model building. It's especially useful for professionals who may not have a deep statistical background.

## Expected Outcomes
- **Efficiency**: Reduces time and effort required for statistical analysis in data science projects.
- **User-Friendly**: Offers a straightforward tool for accurate statistical testing and feature selection.
- **Open-Source Contribution**: Aids educational and professional data science work by contributing to the open-source community.

## Installation

To install this library, you can use the following command:

```bash
pip install git+https://github.com/pranavsai-98/StatDecisionKit.git
import StatDecisionKit as sdk

