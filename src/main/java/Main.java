import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
        Network network = new Network(2, List.of(3), 1);

//        double[] input = new double[5];
//
//        double[] output = network.evaluate(input);
//
//        System.out.println(asList(output));
//
//        List<double[]> activations = network.computeActivations(input);
//
//        for (double[] activation : activations) {
//            System.out.println(asList(activation));
//        }

        dumpGradients(network, new double[] {0, 0}, new double[] {0});
        dumpGradients(network, new double[] {0, 1}, new double[] {0});
        dumpGradients(network, new double[] {1, 0}, new double[] {0});
        dumpGradients(network, new double[] {1, 1}, new double[] {1});
    }

    private static void dumpGradients(Network network, double[] input, double[] ouput)
    {
        List<Layer> layers = network.computeGradients(input, ouput);
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("=== Layer " + i);

            System.out.println("= Weights");
            System.out.println(layers.get(i).getWeights());
            System.out.println("= Biases");
            System.out.println(layers.get(i).getBiases());
        }
    }

    private static List<Double> asList(double[] activation)
    {
        return Arrays.stream(activation).boxed().collect(Collectors.toList());
    }
}
