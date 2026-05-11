package core;

import weka.classifiers.Classifier;

public class AlgorithmConfig {
    public String name;
    public Classifier classifier;
    public DataType type;

    public AlgorithmConfig(String name, Classifier classifier, DataType type) {
        this.name = name;
        this.classifier = classifier;
        this.type = type;
    }
}
