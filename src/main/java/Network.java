import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Network
{
    private List<Layer> layers;

    /**
     * First element in array is amount of inputs to network
     * Last element in array is amount of outputs from network
     */
    public Network(int inputs, List<Integer> hidden, int outputs)
    {
        List<Integer> sizes = new ArrayList<>(hidden);
        sizes.add(outputs);

        layers = new ArrayList<>(sizes.size());

        int previous = inputs;
        for (int size : sizes) {
            layers.add(new Layer(previous, size));
            previous = size;
        }
    }

    public double[] evaluate(double[] input)
    {
        SimpleMatrix nextInput = toMatrix(input);
        for (Layer layer : layers) {
            nextInput = layer.getOutputs(nextInput);
        }

        return NNUtils.toVector(nextInput);
    }

    public void applyDelta(List<Layer> deltas)
    {
        List<Layer> newLayers = new ArrayList<>();

        for (int i = 0; i < layers.size(); i++) {
            SimpleMatrix deltaWeight = deltas.get(i).getWeights();
            SimpleMatrix currentWeight = layers.get(i).getWeights();

            SimpleMatrix deltaBias = deltas.get(i).getBiases();
            SimpleMatrix currentBiases = layers.get(i).getBiases();

            currentWeight.plus(deltaWeight);

            newLayers.add(new Layer(layers.get(i).getNeurons(), currentWeight.minus(deltaWeight), currentBiases.minus(deltaBias)));
        }

        this.layers = newLayers;
    }

    public List<Layer> computeGradients(List<TrainingSample> samples)
    {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("Training samples is empty");
        }

        List<Layer> sum = samples.stream()
                .map(this::computeGradients)
                .reduce(null, (cumulative, gradients) -> cumulative == null ? gradients : addLayerLists(cumulative, gradients));

        return scaleLayerList(sum, 1d / samples.size());
    }

    private List<Layer> computeGradients(TrainingSample sample)
    {
        List<double[]> activations = computeActivations(sample.inputs());

        int currentActivations = activations.size() - 1;

        SimpleMatrix expected = toMatrix(sample.expectedOutputs());
        SimpleMatrix delta = toMatrix(activations.get(currentActivations))
                .minus(expected)
                .scale(2);

        List<Layer> result = new ArrayList<>(layers.size());

        for (int currentLayer = layers.size() - 1; currentLayer >= 0; currentLayer--) {
            SimpleMatrix sigmoidPrime = apply(apply(toMatrix(activations.get(currentActivations)), NNUtils::inverseSigmoid), NNUtils::sigmoidDerivative);
            delta = delta.elementMult(sigmoidPrime);

            result.add(0, new Layer(
                    layers.get(currentLayer).getNeurons(),
                    delta.mult(toMatrix(activations.get(currentActivations - 1)).transpose()),
                    delta));

            delta = layers.get(currentLayer).getWeights()
                    .transpose()
                    .mult(delta);

            currentActivations--;
        }

        return result;
    }

    private List<double[]> computeActivations(double[] input)
    {
        List<double[]> result = new ArrayList<>(layers.size());

        SimpleMatrix activations = toMatrix(input);
        result.add(NNUtils.toVector(activations));
        for (Layer layer : layers) {
            activations = layer.getOutputs(activations);
            result.add(NNUtils.toVector(activations));
        }

        return result;
    }

    public static List<Layer> scaleLayerList(List<Layer> layers, double scale)
    {
        List<Layer> result = new ArrayList<>(layers.size());

        for (Layer layer : layers) {
            SimpleMatrix weights = layer.getWeights().scale(scale);
            SimpleMatrix biases = layer.getBiases().scale(scale);
            result.add(new Layer(layer.getNeurons(), weights, biases));
        }
        return result;
    }

    private static List<Layer> addLayerLists(List<Layer> layer1, List<Layer> layer2)
    {
        if (layer1.size() != layer2.size()) {
            throw new IllegalArgumentException("Layer list must have the same size");
        }

        List<Layer> result = new ArrayList<>(layer1.size());

        for (int i = 0; i < layer1.size(); i++) {
            SimpleMatrix weights = layer1.get(i).getWeights().plus(layer2.get(i).getWeights());
            SimpleMatrix biases = layer1.get(i).getBiases().plus(layer2.get(i).getBiases());
            result.add(new Layer(layer1.get(i).getNeurons(), weights, biases));
        }

        return result;
    }

    private static SimpleMatrix apply(SimpleMatrix matrix, Function<Double, Double> function)
    {
        SimpleMatrix result = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int row = 0; row < matrix.numRows(); row++) {
            for (int column = 0; column < result.numCols(); column++) {
                result.set(row, column, function.apply(matrix.get(row, column)));
            }
        }

        return result;
    }

    private static SimpleMatrix toMatrix(double[] values)
    {
        return new SimpleMatrix(values.length, 1, true, values);
    }



    public static void printLayers(List<Layer> layers)
    {
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("=== Layer " + i);

            System.out.println("= Weights");
            System.out.println(layers.get(i).getWeights());
            System.out.println("= Biases");
            System.out.println(layers.get(i).getBiases());
        }
    }

    public void dump()
    {
        printLayers(layers);
    }

    public static void train(Network network, List<TrainingSample> andGateSamples)
    {
        for (int i = 0; i < 10000; i++) {
            List<Layer> layers = network.computeGradients(andGateSamples);

            Network.scaleLayerList(layers, 0.15);
            network.applyDelta(layers);
        }
    }

    public void train(List<TrainingSample> andGateSamples)
    {
        train(this, andGateSamples);
    }
}
