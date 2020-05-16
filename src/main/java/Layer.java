import org.ejml.simple.SimpleMatrix;

import java.util.concurrent.ThreadLocalRandom;

public class Layer
{
    private final SimpleMatrix weightMatrix;
    private final SimpleMatrix biasMatrix;
    private final int neurons;

    public Layer(int inputs, int neurons)
    {
        this.neurons = neurons;

        weightMatrix = new SimpleMatrix(getRandomArray(neurons, inputs));
        biasMatrix = new SimpleMatrix(getRandomArray(neurons, 1));
    }

    public SimpleMatrix getOutputs(SimpleMatrix inputs)
    {
        SimpleMatrix matrix = weightMatrix.mult(inputs).plus(biasMatrix);
        for (int i = 0; i < neurons; i++) {
            double sigmoidVal = NNMath.sigmoid(matrix.get(i, 0));
            matrix.set(i, 0, sigmoidVal);
        }
        return matrix;
    }

    private static double[][] getRandomArray(int rows, int cols)
    {
        double[][] arr = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[i][j] = ThreadLocalRandom.current().nextDouble()*2d - 1d;
            }
        }
        return arr;
    }
}
