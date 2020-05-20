package org.marcos.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public final class TrainingConfigs
{
    private TrainingConfigs()
    {
        throw new AssertionError();
    }

    public static Network createAndTrainAND()
    {
        List<TrainingSample> samples = List.of(
                new TrainingSample(new double[] {0, 0}, new double[] {0}),
                new TrainingSample(new double[] {0, 1}, new double[] {0}),
                new TrainingSample(new double[] {1, 0}, new double[] {0}),
                new TrainingSample(new double[] {1, 1}, new double[] {1})
        );

        Network network = new Network(2, List.of(), 1);

        network.train(samples, 10000, 0.1);

        network.dump();
        for (TrainingSample sample : samples) {
//            System.out.println(NNUtils.asList(NNUtils.roundAll(network.evaluate(sample.inputs()))));
            System.out.println(NNUtils.asList(network.evaluate(sample.inputs())));
        }
        return network;
    }

    public static Network createAndTrain3AND()
    {
        List<TrainingSample> samples = List.of(
                new TrainingSample(new double[] {0, 0, 0}, new double[] {0}),
                new TrainingSample(new double[] {0, 0, 1}, new double[] {0}),
                new TrainingSample(new double[] {0, 1, 0}, new double[] {0}),
                new TrainingSample(new double[] {0, 1, 1}, new double[] {0}),
                new TrainingSample(new double[] {1, 0, 0}, new double[] {0}),
                new TrainingSample(new double[] {1, 0, 1}, new double[] {0}),
                new TrainingSample(new double[] {1, 1, 0}, new double[] {0}),
                new TrainingSample(new double[] {1, 1, 1}, new double[] {1})
        );

        Network network = new Network(3, List.of(), 1);

            network.train(samples, 2000, 0.1);

        network.dump();
        for (TrainingSample sample : samples) {
            System.out.println(NNUtils.asList(NNUtils.roundAll(network.evaluate(sample.inputs()))));
        }
        return network;
    }

    public static Network createAndTrainXSquared()
    {
        List<TrainingSample> samples = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            double value = ThreadLocalRandom.current().nextDouble();
            samples.add(new TrainingSample(new double[] {value}, new double[] {value * value}));
        }

        Network network = new Network(1, List.of(10,8), 1);

        network.train(samples, 10000, 0.5);

        network.dump();
        System.out.println(NNUtils.asList(network.evaluate(new double[] {0.3})));
        return network;
    }
}
