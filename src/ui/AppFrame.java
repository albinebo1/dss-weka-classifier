package ui;

import javax.swing.*;
import java.awt.*;

public class AppFrame extends JFrame {

    public AppFrame() {
        setTitle("Applications of Decision Support Systems Classification Project");
        setSize(900, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JTabbedPane tabs = new JTabbedPane();

        TrainingPanel training = new TrainingPanel();
        DiscoverPanel discover = new DiscoverPanel(training.core);
        training.setDiscoverPanel(discover);

        tabs.addTab("Training", training);
        tabs.addTab("Discover", discover);

        add(tabs, BorderLayout.CENTER);
    }
}
