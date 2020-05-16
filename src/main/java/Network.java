import org.ejml.simple.SimpleMatrix;

public class Network
{
    private final Layer[] layers;

    /**
     * First element in array is amount of inputs to network
     * Last element in array is amount of outputs from network
     */
    public Network(int[] nodesPerLayer)
    {
        this.layers = new Layer[nodesPerLayer.length - 1];

        for (int i = 0; i < nodesPerLayer.length - 1; i++) {
            this.layers[i] = new Layer(nodesPerLayer[i], nodesPerLayer[i + 1]);    //i-> inputs, i+1 neurons for following layer
        }
    }

    public SimpleMatrix feed(SimpleMatrix InitialInput)
    {
        SimpleMatrix nextInput = InitialInput;
        for (Layer currLayer : layers) {
                nextInput = currLayer.getOutputs(nextInput);
        }
        return nextInput;
    }
}
