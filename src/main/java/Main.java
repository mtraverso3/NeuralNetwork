import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
        List<TrainingSample> andGateSamples = List.of(
                new TrainingSample(new double[] {0, 0}, new double[] {0}),
                new TrainingSample(new double[] {0, 1}, new double[] {0}),
                new TrainingSample(new double[] {1, 0}, new double[] {0}),
                new TrainingSample(new double[] {1, 1}, new double[] {1}));

        Network network = new Network(2, List.of(), 1);

        for (int i = 0; i < 1000; i++) {
            List<Layer> layers = network.computeGradients(andGateSamples);

            Network.scaleLayerList(layers, 0.01);
            network.addDelta(layers);
        }

        network.dump();
        for (TrainingSample sample :
                andGateSamples) {
            System.out.println(asList(NNMath.roundAll(network.evaluate(sample.inputs()))));
        }
    }

    private static List<Double> asList(double[] activation)
    {
        return Arrays.stream(activation).boxed().collect(Collectors.toList());
    }
}
