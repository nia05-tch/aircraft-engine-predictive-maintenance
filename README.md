# Aircraft Engine Predictive Maintenance

Predicting turbofan engine failures using NASA C-MAPSS dataset with sensor data analysis and Random Forest classification.

## Problem Statement

Airlines need to predict engine failures before they happen to:
- Prevent in-flight emergencies
- Reduce maintenance costs
- Optimize aircraft availability

## Approach

1. **Data Exploration**: Analyzed 21 sensors across 100 engines
2. **Feature Discovery**: Identified sensor3 and sensor4 as strongest predictors through comparative analysis
3. **Class Imbalance Handling**: Used class weights to prioritize failure detection
4. **Model**: Random Forest with 100 estimators

## Key Results

- **89% Recall**: Catches 89% of engine failures
- **92% Overall Accuracy**: Strong general performance
- **Trade-off**: 32% false alarm rate (acceptable for safety-critical applications)

## Technical Stack

- Python, pandas, scikit-learn, matplotlib, seaborn
- NASA C-MAPSS Turbofan Engine Degradation Dataset
- Random Forest with balanced class weights

## Real-World Impact

In aerospace maintenance, missing a failure is far more costly than unnecessary inspections. This model prioritizes catching failures (high recall) over minimizing false alarms.

---

**Author**: Nia Aracheva  
**Focus**: Autonomous Systems & Aerospace AI  
**Date**: January 2026