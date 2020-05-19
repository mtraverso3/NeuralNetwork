public final class NNMath
{
    // Suppress default constructor for noninstantiability
    private NNMath()
    {
        throw new AssertionError();
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

    public static double tanh(double x)
    {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x)
    {
        return 1 - Math.tanh(x) * Math.tanh(x);
    }

    public static double[] roundAll(double[] array){
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            double value = array[i];
            result[i] = Math.round(value);
        }
        return result;
    }
}
