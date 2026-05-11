package core;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.*;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.*;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;

//ab
import java.util.*;

public class WekaCore {

    public Instances originalData; // dataset as loaded
    public Instances numericData; // dataset converted to numeric - nominalToBinary + normalized
    public Instances nominalData; // dataset converted to nominal - discretize

    // filters used to build numericData / nominalData (same preprocessing during prediction)
    public Filter numericFilter;
    public Filter nominalFilter;

    public Classifier bestClassifier;
    public String bestName;
    public double bestCorrect;
    public DataType bestType;   // NUMERIC / NOMINAL / ANY

    public ArrayList<AlgorithmConfig> configs;

    public WekaCore() {
        buildConfigs();
    }

    public void loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        originalData = source.getDataSet();

        //set class index - required by weka
        if (originalData.classIndex() == -1) {
            originalData.setClassIndex(originalData.numAttributes() - 1);
        }

        // preparing transformed datasets
        prepareNumericData();
        prepareNominalData();
    }

    // create numericData + numericFilter (NominalToBinary + Normalize)
    // Basically a numeric dataset for KNN, logistic regression, SVM and multilayer perceptron
    // convert nominal attributes to dummy ones using NominalToBinary
    // normalize numeric attributes
    private void prepareNumericData() throws Exception {
        Instances data = new Instances(originalData);

        boolean allNumeric = true;
        for (int i = 0; i < data.numAttributes(); i++) { // check if all attributes are numeric
            if (i != data.classIndex() && !data.attribute(i).isNumeric()) {
                allNumeric = false;
                break;
            }
        }

        if (!allNumeric) {
            // need nominal->binary AND normalize
            NominalToBinary ntb = new NominalToBinary();
            Normalize norm = new Normalize();

            MultiFilter mf = new MultiFilter();
            mf.setFilters(new Filter[]{ntb, norm});
            mf.setInputFormat(data);

            numericFilter = mf;
            numericData = Filter.useFilter(data, numericFilter);
        } else {
            // only normalize
            Normalize norm = new Normalize();
            norm.setInputFormat(data);
            numericFilter = norm;
            numericData = Filter.useFilter(data, numericFilter);
        }
    }

    // create nominalData + nominalFilter (Discretize)
    // needed for naive-bayes
    private void prepareNominalData() throws Exception {
        Instances data = new Instances(originalData);

        boolean allNominal = true;
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex() && !data.attribute(i).isNominal()) {
                allNominal = false;
                break;
            }
        }

        if (!allNominal) {
            Discretize disc = new Discretize();
            disc.setInputFormat(data);
            nominalFilter = disc;
            nominalData = Filter.useFilter(data, nominalFilter);
        } else {
            nominalFilter = null; // no transform
            nominalData = data;
        }
    }

    private void buildConfigs() { // algorithm configurations with their respective needed data types
        configs = new ArrayList<>();

        configs.add(new AlgorithmConfig("NaiveBayes", new NaiveBayes(), DataType.NOMINAL));
        configs.add(new AlgorithmConfig("Logistic", new Logistic(), DataType.NUMERIC));
        configs.add(new AlgorithmConfig("IBk k=1", new IBk(1), DataType.NUMERIC));
        configs.add(new AlgorithmConfig("IBk k=3", new IBk(3), DataType.NUMERIC));
        configs.add(new AlgorithmConfig("J48", new J48(), DataType.ANY));
        configs.add(new AlgorithmConfig("RandomForest 100 trees", new RandomForest(), DataType.ANY));

        RandomForest rf50 = new RandomForest();
        rf50.setNumIterations(50);
        configs.add(new AlgorithmConfig("RandomForest 50 trees", rf50, DataType.ANY));
        configs.add(new AlgorithmConfig("RandomTree", new RandomTree(), DataType.ANY));
        configs.add(new AlgorithmConfig("MultilayerPerceptron", new MultilayerPerceptron(), DataType.NUMERIC));
        configs.add(new AlgorithmConfig("SVM (SMO)", new SMO(), DataType.NUMERIC));
    }

    public ArrayList<String[]> runAll() throws Exception { // run evaluation for all algorithms
                                                           // using cross validation for all 10
        bestCorrect = -1;
        bestClassifier = null;
        bestName = null;
        bestType = null;

        ArrayList<String[]> table = new ArrayList<>();

        for (AlgorithmConfig cfg : configs) {

            Classifier classifierToEval;
            Instances evalData = originalData;

            // apply correct preprocessing depending on algorithm
            if (cfg.type == DataType.NUMERIC) {
                FilteredClassifier fc = new FilteredClassifier();
                fc.setClassifier(cfg.classifier);
                fc.setFilter(numericFilter);
                classifierToEval = fc;

            } else if (cfg.type == DataType.NOMINAL) {
                FilteredClassifier fc = new FilteredClassifier();
                fc.setClassifier(cfg.classifier);
                fc.setFilter(nominalFilter);
                classifierToEval = fc;

            } else {
                // trees ; no preprocessing required
                classifierToEval = cfg.classifier;
            }

            Evaluation eval = new Evaluation(originalData);
            eval.crossValidateModel(classifierToEval, originalData, 10, new Random(1));

            double correct = eval.correct();

            table.add(new String[]{
                    cfg.name,
                    String.format("%.2f%%", eval.pctCorrect()),
                    correct + ""
            });

            if (correct > bestCorrect) { // keep track of the best classifier
                bestCorrect = correct;
                bestName = cfg.name;
                bestType = cfg.type;

                bestClassifier = AbstractClassifier.makeCopy(classifierToEval);
                bestClassifier.buildClassifier(originalData);
            }
        }

        return table;
    }


    // userValues are in ORIGINAL attribute order (from originalData)
    public String classify(ArrayList<Object> userValues) throws Exception { // predict class of a new instance
        // use same preprocessing as we did during training
        // Build a 1-row dataset using original structure
        Instances base = new Instances(originalData, 0); // same header, class index set
        DenseInstance origInst = new DenseInstance(base.numAttributes());
        origInst.setDataset(base);

        int uiIndex = 0;
        for (int i = 0; i < base.numAttributes(); i++) {
            if (i == base.classIndex()) continue;

            Attribute a = base.attribute(i);
            Object val = userValues.get(uiIndex++);

            if (a.isNominal()) {
                origInst.setValue(a, val.toString());
            } else {
                origInst.setValue(a, Double.parseDouble(val.toString()));
            }
        }
        base.add(origInst);

        // apply the same preprocessing used by the best classifier
        Instances transformed;

        switch (bestType) {
            case NUMERIC:
                if (numericFilter != null) {
                    transformed = Filter.useFilter(base, numericFilter);
                } else {
                    transformed = base;
                }
                break;

            case NOMINAL:
                if (nominalFilter != null) {
                    transformed = Filter.useFilter(base, nominalFilter);
                } else {
                    transformed = base;
                }
                break;

            case ANY:
            default:
                transformed = base;
                break;
        }

        Instance finalInst = transformed.instance(0);

        double[] probs = bestClassifier.distributionForInstance(finalInst);
        Attribute classAttr = transformed.classAttribute();

        StringBuilder sb = new StringBuilder();

        // Find predicted class
        int bestIndex = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[bestIndex]) {
                bestIndex = i;
            }
        }

        sb.append("Prediction: ")
                .append(classAttr.value(bestIndex))
                .append("\n");

        // Append probabilities
        for (int i = 0; i < probs.length; i++) {
            sb.append(classAttr.value(i))
                    .append(": ")
                    .append(String.format("%.2f", probs[i]))
                    .append("\n");
        }

        return sb.toString();
        // classify
//        double pred = bestClassifier.classifyInstance(finalInst);
//        return transformed.classAttribute().value((int) pred);
    }
}
