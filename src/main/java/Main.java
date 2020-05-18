import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
        Network network = new Network(2, List.of(1), 1);

        for (int i = 0; i < 1; i++) {
            List<Layer> layers = network.computeGradients(List.of(
                    new TrainingSample(new double[] {0, 0}, new double[] {0}),
                    new TrainingSample(new double[] {0, 1}, new double[] {0}),
                    new TrainingSample(new double[] {1, 0}, new double[] {0}),
                    new TrainingSample(new double[] {1, 1}, new double[] {1})));

            Network.scaleLayerList(layers, 0.01);
            network.addDelta(layers);
        }

        network.dump();

        System.out.println(asList(network.evaluate(new double[] {0, 0})));
        System.out.println(asList(network.evaluate(new double[] {0, 1})));
        System.out.println(asList(network.evaluate(new double[] {1, 0})));
        System.out.println(asList(network.evaluate(new double[] {1, 1})));
    }

//    private static void dumpGradients(Network network, double[] input, double[] output)
//    {
//        List<Layer> layers = network.computeGradients(input, output);
//        for (int i = 0; i < layers.size(); i++) {
//            System.out.println("=== Layer " + i);
//
//            System.out.println("= Weights");
//            System.out.println(layers.get(i).getWeights());
//            System.out.println("= Biases");
//            System.out.println(layers.get(i).getBiases());
//        }
//    }

    private static List<Double> asList(double[] activation)
    {
        return Arrays.stream(activation).boxed().collect(Collectors.toList());
    }
}
