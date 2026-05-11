# DSS WEKA Classifier

A Java desktop application which uses the WEKA library to evaluate multiple classification algorithms on a user-selected dataset and predicts the class of a new instance through a Swing interface.

## Features

- Load `.arff` or `.csv` datasets from the interface
- Evaluate 10 classification approaches:
  - Naive Bayes
  - Logistic Regression
  - IBk k=1
  - IBk k=3
  - J48
  - Random Forest 100 trees
  - Random Forest 50 trees
  - Random Tree
  - Multilayer Perceptron
  - SVM / SMO
- Shows accuracy and number of correctly classified instances
- Selects the best-performing algorithm
- Creates input fields dynamically:
  - Dropdowns for nominal attributes
  - Text fields for numeric attributes
- Predicts the class of a new user-entered instance

## Preprocessing

The application applies the required preprocessing depending on the algorithm:

- Nominal to numeric conversion using `NominalToBinary`
- Numeric to nominal conversion using `Discretize`
- Normalization for numeric datasets

## Requirements

- Java
- WEKA library / `weka.jar`
- IntelliJ IDEA or another Java IDE

## How to Run

1. Clone the repository.
2. Add `weka.jar` to the project libraries.
3. Run `Main.java`.
4. Load a dataset from the **Training** tab.
5. Click **Run Evaluation**.
6. Go to the **Discover** tab, enter values, and click **Discover** to predict the class.

## Project Structure

```text
src/
├── Main.java
├── core/
│   ├── AlgorithmConfig.java
│   ├── DataType.java
│   └── WekaCore.java
└── ui/
    ├── AppFrame.java
    ├── DiscoverPanel.java
    └── TrainingPanel.java
