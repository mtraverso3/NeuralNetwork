package org.marcos.neuralnetwork;

public class Main
{
    public static void main(String[] args)
    {
//        Network network = TrainingConfigs.createAndTrainAND();
//        Network network = TrainingConfigs.createAndTrain3AND();
        long start = System.nanoTime();
        Network network = TrainingConfigs.createAndTrainXSquared();
        System.out.println((System.nanoTime() - start) / 1e9);

    }
}
