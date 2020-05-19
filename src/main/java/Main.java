import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
//        trainAnd();
        trainAnd3();
//        trainXSquared();
    }

    private static void trainXSquared()
    {
        List<TrainingSample> samples = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            double value = ThreadLocalRandom.current().nextDouble();
            samples.add(new TrainingSample(new double[] {value}, new double[] {value * value}));
        }

        Network network = new Network(1, List.of(3), 1);

        train(samples, network);

        network.dump();

        System.out.println(asList(network.evaluate(new double[] { 0.3 })));
    }

    private static void trainAnd3()
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

        Network network = new Network(3, List.of(2, 2), 1);

        train(samples, network);

        network.dump();
        for (TrainingSample sample : samples) {
            System.out.println(asList(NNMath.roundAll(network.evaluate(sample.inputs()))));
        }
    }

    private static void trainAnd()
    {
        List<TrainingSample> andGateSamples = List.of(
                new TrainingSample(new double[] {0, 0}, new double[] {1}),
                new TrainingSample(new double[] {0, 1}, new double[] {1}),
                new TrainingSample(new double[] {1, 0}, new double[] {1}),
                new TrainingSample(new double[] {1, 1}, new double[] {0})
        );

        Network network = new Network(2, List.of(2, 2), 1);

        train(andGateSamples, network);

        network.dump();
        for (TrainingSample sample : andGateSamples) {
            System.out.println(asList(NNMath.roundAll(network.evaluate(sample.inputs()))));
        }
    }

    private static void train(List<TrainingSample> andGateSamples, Network network)
    {
        for (int i = 0; i < 100000; i++) {
            List<Layer> layers = network.computeGradients(andGateSamples);

            Network.scaleLayerList(layers, 0.01);
            network.applyDelta(layers);
        }
    }

    private static List<Double> asList(double[] activation)
    {
        return Arrays.stream(activation).boxed().collect(Collectors.toList());
    }
}
