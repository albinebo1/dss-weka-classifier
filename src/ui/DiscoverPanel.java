package ui;

import core.WekaCore;
import weka.core.Attribute;
import weka.core.Instances;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
//a
public class DiscoverPanel extends JPanel {

    WekaCore core; // reference to the core weka logic
    JPanel fieldsPanel; // panel that holds the input fields
    JButton predictBtn; // predict button
    JLabel resultLabel; // result label

    //structure used for UI
    Instances structure;

    public DiscoverPanel(WekaCore core) {
        this.core = core;
        setLayout(new BorderLayout());

        fieldsPanel = new JPanel();
        fieldsPanel.setLayout(new GridLayout(0, 2, 10, 10));
        add(fieldsPanel, BorderLayout.CENTER);

        predictBtn = new JButton("Discover");
        resultLabel = new JLabel("Prediction: -");

        JPanel bottom = new JPanel();
        bottom.add(predictBtn);
        bottom.add(resultLabel);
        add(bottom, BorderLayout.SOUTH);

        predictBtn.addActionListener(e -> doPredict());
    }

    // Rebuild inputs with the dataset structure actually used for training
    public void rebuildInputs(Instances structure) {
        this.structure = structure;

        fieldsPanel.removeAll();

        if (structure == null) {
            fieldsPanel.add(new JLabel("Load dataset first"));
            fieldsPanel.add(new JLabel());
            revalidate();
            repaint();
            return;
        }

        for (int i = 0; i < structure.numAttributes(); i++) { // go column by column
            if (i == structure.classIndex()) continue; // skip class attribute - thats the one we want to predict

            Attribute a = structure.attribute(i);

            fieldsPanel.add(new JLabel(a.name())); // add text label to grid panel for user to know what they're entering

            if (a.isNominal()) { // if nominal add dropdown
                JComboBox<String> box = new JComboBox<>();
                for (int j = 0; j < a.numValues(); j++)
                    box.addItem(a.value(j));
                fieldsPanel.add(box);
            } else { // otherwise add textbox
                fieldsPanel.add(new JTextField());
            }
        }

        revalidate();
        repaint();
    }

    private void doPredict() { // classifies after user inputs his own instance
        try {
            if (structure == null) {
                JOptionPane.showMessageDialog(null, "Train a model first!");
                return; // when dataset is not loaded
            }           // or when loaded but not evaluated yet

            ArrayList<Object> values = new ArrayList<>();

            int componentIndex = 1; // inside fieldsPanel components go like 0 - JLabel, 1 - Input, 2 - JLabel, 3 - Input, ...
                                    // so we start from index 1 because that is the input field
            for (int i = 0; i < structure.numAttributes(); i++) {
                if (i == structure.classIndex()) continue; //skip class attribute

                Component comp = fieldsPanel.getComponent(componentIndex); //-either JComboBox - nominal or JTextField - numeric

                if (comp instanceof JComboBox) { // if nominal choose and store value
                    JComboBox<?> box = (JComboBox<?>) comp;
                    values.add(box.getSelectedItem());
                } else { // if numeric store text from textbox
                    JTextField txt = (JTextField) comp;
                    values.add(txt.getText());
                }

                componentIndex += 2; // jump to next input field
            }

            String pred = core.classify(values); // call classifier
                        // core builds a weka instance, applies correct filters
                        // runs the best classifier, returns predicted class label
            resultLabel.setText("<html>" + pred.replace("\n", "<br>") + "</html>");


        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, "Error predicting!");
            e.printStackTrace();
        }
    }
}
