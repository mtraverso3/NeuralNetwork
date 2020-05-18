import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
        Network network = new Network(2, List.of(3), 1);
    }

    private static void train(Network network, double[][] inputData, double[][] outputData, int amount)
    {

        List<Layer> prevGradients = network.computeGradients(inputData[0], outputData[0]);
        for (int i = 1; i < inputData.length; i++) {
            List<Layer> gradients = network.computeGradients(inputData[i], outputData[i]);
            gradients = addLayerLists(gradients, prevGradients);
            prevGradients = gradients;
        }
        scaleLayerList(prevGradients, 1d / inputData.length);
    }

    private static List<Layer> scaleLayerList(List<Layer> layer1, double scale)
    {
        List<Layer> OutputLayer = new ArrayList<Layer>(layer1.size());

        for (int i = 0; i < layer1.size(); i++) {
            SimpleMatrix weights = layer1.get(i).getWeights().scale(scale);
            SimpleMatrix biases = layer1.get(i).getBiases().scale(scale);
            OutputLayer.set(i, new Layer(layer1.get(i).getNeurons(), weights, biases));
        }
        return OutputLayer;
    }

    private static List<Layer> addLayerLists(List<Layer> layer1, List<Layer> layer2)
    {
        List<Layer> OutputLayer = new ArrayList<Layer>(layer1.size());

        for (int i = 0; i < layer1.size(); i++) {
            SimpleMatrix weights = layer1.get(i).getWeights().plus(layer2.get(i).getWeights());
            SimpleMatrix biases = layer1.get(i).getBiases().plus(layer2.get(i).getBiases());
            OutputLayer.set(i, new Layer(layer1.get(i).getNeurons(), weights, biases));
        }
        return OutputLayer;
    }

    private static void dumpGradients(Network network, double[] input, double[] output)
    {
        List<Layer> layers = network.computeGradients(input, output);
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
