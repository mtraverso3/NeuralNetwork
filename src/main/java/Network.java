import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Network
{
    private final List<Layer> layers;

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

        return toVector(nextInput);
    }

    private static SimpleMatrix toMatrix(double[] values)
    {
        return new SimpleMatrix(values.length, 1, true, values);
    }

    /**
     * Extract vector of doubles from single-column matrix
     */
    private static double[] toVector(SimpleMatrix matrix)
    {
        double[] output = new double[matrix.getNumElements()];
        for (int i = 0; i < output.length; i++) {
            output[i] = matrix.get(i, 0);
        }
        return output;
    }

    public List<Layer> computeGradients(double[] input, double[] expectedOutputs)
    {
        List<double[]> activations = computeActivations(input);

        int current = activations.size() - 1;

        SimpleMatrix activation = toMatrix(activations.get(current));
        SimpleMatrix expected = toMatrix(expectedOutputs);
        SimpleMatrix delta = activation
                .minus(expected)
                .scale(2);

        List<Layer> result = new ArrayList<>(layers.size());

        while (current > 0) {
            SimpleMatrix z = apply(activation, NNMath::inverseSigmoid);
            SimpleMatrix sigmoidPrime = apply(z, NNMath::sigmoidDerivative);

            delta = delta.mult(sigmoidPrime.transpose()).transpose();

            double[] previousLayerActivations = activations.get(current - 1);
            result.add(0, new Layer(
                    layers.get(current - 1).getNeurons(),
                    delta.mult(toMatrix(previousLayerActivations).transpose()),
                    delta));

            activation = toMatrix(previousLayerActivations);
            current--;
        }

        return result;
    }

    private SimpleMatrix apply(SimpleMatrix matrix, Function<Double, Double> function)
    {
        SimpleMatrix result = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int row = 0; row < matrix.numRows(); row++) {
            for (int column = 0; column < result.numCols(); column++) {
                result.set(row, column, function.apply(matrix.get(row, column)));
            }
        }

        return result;
    }

    public List<double[]> computeActivations(double[] input)
    {
        List<double[]> result = new ArrayList<>(layers.size());

        SimpleMatrix activations = toMatrix(input);
        result.add(toVector(activations));
        for (Layer layer : layers) {
            activations = layer.getOutputs(activations);
            result.add(toVector(activations));
        }

        return result;
    }
}
