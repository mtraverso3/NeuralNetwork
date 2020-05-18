import org.ejml.simple.SimpleMatrix;

public class Layer
{
    private final SimpleMatrix weights;
    private final SimpleMatrix biases;
    private final int neurons;

    public Layer(int inputs, int neurons)
    {
        this.neurons = neurons;

        weights = new SimpleMatrix(neurons, inputs, true, new double[inputs * neurons]);
        biases = new SimpleMatrix(neurons, 1, true, new double[neurons]);
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
            double sigmoidVal = NNMath.sigmoid(matrix.get(i, 0));
            matrix.set(i, 0, sigmoidVal);
        }
        return matrix;
    }
}
