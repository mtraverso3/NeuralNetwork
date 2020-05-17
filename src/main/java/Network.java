import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public class Network
{
    private final ArrayList<Layer> layers;

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

    public SimpleMatrix feed(SimpleMatrix InitialInput)
    {
        SimpleMatrix nextInput = InitialInput;
        for (Layer layer : layers) {
            nextInput = layer.getOutputs(nextInput);
        }
        return nextInput;
    }
}
