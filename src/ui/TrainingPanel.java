package ui;

import core.WekaCore;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class TrainingPanel extends JPanel {

    public WekaCore core = new WekaCore();
    JTable table;
    JLabel bestLabel;
    public DiscoverPanel discoverPanel;

    public void setDiscoverPanel(DiscoverPanel p) {
        this.discoverPanel = p;
    }

    public TrainingPanel() {
        setLayout(new BorderLayout());

        // create buttons to load and evaluate dataset
        JPanel top = new JPanel();
        JButton loadBtn = new JButton("Load Dataset");
        JButton trainBtn = new JButton("Run Evaluation");

        top.add(loadBtn);
        top.add(trainBtn);

        add(top, BorderLayout.NORTH);

        table = new JTable();
        add(new JScrollPane(table), BorderLayout.CENTER);

        bestLabel = new JLabel("Best: -");
        bestLabel.setFont(new Font("Arial", Font.BOLD, 16));
        add(bestLabel, BorderLayout.SOUTH);

        loadBtn.addActionListener(e -> { // choose dataset from file
            try {
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(
                        new javax.swing.filechooser.FileNameExtensionFilter(
                                "Datasets (*.arff, *.csv)", "arff", "csv"
                        )
                );

                if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                    core.loadData(fc.getSelectedFile().getAbsolutePath());
                    JOptionPane.showMessageDialog(null, "Dataset Loaded!");

                    //reset inputs using the original structure until training occurs
                    discoverPanel.rebuildInputs(core.originalData);
                }
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(null, "Error loading dataset.");
            }
        });

        trainBtn.addActionListener(e -> {
            try {
                ArrayList<String[]> res = core.runAll();

                //rebuild using the structure the best classifier was trained on
                discoverPanel.rebuildInputs(core.originalData);

                DefaultTableModel model = new DefaultTableModel();
                model.addColumn("Algorithm");
                model.addColumn("Accuracy (%)");
                model.addColumn("Correct");

                for (String[] row : res) {
                    model.addRow(row);
                }

                table.setModel(model);

                bestLabel.setText("BEST: " + core.bestName +
                        " (Correct: " + core.bestCorrect + ")");

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });
    }
}
