package org.marcos.neuralnetwork;

import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public final class NNUtils
{
    // Suppress default constructor for noninstantiability
    private NNUtils()
    {
        throw new AssertionError();
    }

    public static List<Double> asList(double[] activation)
    {
        return Arrays.stream(activation).boxed().collect(Collectors.toList());
    }

    /**
     * Extract vector of doubles from single-column matrix
     */
    public static double[] toVector(SimpleMatrix matrix)
    {
        double[] output = new double[matrix.getNumElements()];
        for (int i = 0; i < output.length; i++) {
            output[i] = matrix.get(i, 0);
        }
        return output;
    }

    public static double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static double inverseSigmoid(double x)
    {
        return -Math.log(1 / x - 1);
    }

    public static double[] roundAll(double[] array)
    {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            double value = array[i];
            result[i] = Math.round(value);
        }
        return result;
    }
}
