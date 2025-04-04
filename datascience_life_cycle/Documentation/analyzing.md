# Exploratory Data Analysis (EDA)

## Introduction

Exploratory Data Analysis (EDA) helps us understand the data and prepare it for modeling. It involves analyzing features, relationships, and identifying problems within the data. In this lesson, we’ll use a Kaggle dataset with word counts from anonymous emails, applying EDA using Python and Pandas.

## What is EDA?

EDA helps answer questions like:
- Do we have enough data to solve the problem?
- Is the data of good quality?
- Should the goals be redefined based on new insights?

### Key Techniques in EDA

- **Data Profiling & Descriptive Statistics**: Use Pandas' `describe()` function to get an overview (count, mean, min, max, standard deviation) of the dataset and assess its quality.
  
- **Sampling**: Sampling helps explore large datasets efficiently. Use the `sample()` function in Pandas for random data samples to make general conclusions.

- **Querying**: The `query()` function helps focus on specific data segments based on conditions, allowing targeted insights.

## Visualizations

Visualizations help identify patterns, relationships, and issues early on. They also make it easier to communicate insights to others. 

## Identifying Inconsistencies

Check for missing or inconsistent values using `isna()` or `isnull()`. Understanding why data is missing helps determine how to address these issues (e.g., filling, removing, or correcting data).
