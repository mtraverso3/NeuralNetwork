package org.marcos.neuralnetwork;

import org.ejml.simple.SimpleMatrix;

import static java.lang.String.format;

public class Layer
{
    private final SimpleMatrix weights;
    private final SimpleMatrix biases;
    private final int neurons;

    public Layer(int inputs, int outputs)
    {
        this.neurons = outputs;

        weights = new SimpleMatrix(outputs, inputs, true, new double[inputs * outputs]);
        biases = new SimpleMatrix(outputs, 1, true, new double[outputs]);
    }

    public Layer(int neurons, SimpleMatrix weights, SimpleMatrix biases)
    {
        this.neurons = neurons;
        this.weights = weights;
        this.biases = biases;
    }

    public int getNeurons()
    {
        return neurons;
    }

    public SimpleMatrix getWeights()
    {
        return weights;
    }

    public SimpleMatrix getBiases()
    {
        return biases;
    }

    public SimpleMatrix getOutputs(SimpleMatrix inputs)
    {
        SimpleMatrix matrix = weights.mult(inputs).plus(biases);
        for (int i = 0; i < neurons; i++) {
            double sigmoidVal = NNUtils.sigmoid(matrix.get(i, 0));
            matrix.set(i, 0, sigmoidVal);
        }
        return matrix;
    }

    @Override
    public String toString()
    {
        return format("[inputs=%s, outputs=%s]: w=%sx%x, b=%sx%x",
                weights.numCols(),
                biases.numRows(),
                weights.numRows(),
                weights.numCols(),
                biases.numRows(),
                biases.numCols());
    }
}
