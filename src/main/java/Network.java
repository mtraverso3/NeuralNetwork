import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;

public class Network
{
    private final ArrayList<Layer> layers;

    /**
     * First element in array is amount of inputs to network
     * Last element in array is amount of outputs from network
     */
    public Network(int inputs, ArrayList<Integer> hidden, int outputs)
    {
        this.layers = new ArrayList<>(hidden.size() + 1);

        if (hidden.size() == 0) {
            this.layers.add(new Layer(inputs, outputs));
        }
        else {
            this.layers.add(new Layer(inputs, hidden.get(0)));

            for (int i = 0; i < hidden.size() - 1; i++) {
                this.layers.add(new Layer(hidden.get(i), hidden.get(i + 1)));
            }
            this.layers.add(new Layer(hidden.get(hidden.size() - 1), outputs));
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
