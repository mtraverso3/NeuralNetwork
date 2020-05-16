import org.ejml.simple.SimpleMatrix;

public class Main
{
    public static void main(String[] args)
    {
        SimpleMatrix inputMatrix = new SimpleMatrix(2, 1);
        inputMatrix.fill(1);

        Network network = new Network(new int[] {2, 3});
        SimpleMatrix outputMatrix = network.feed(inputMatrix);

        System.out.println(inputMatrix.toString());
        System.out.println(outputMatrix.toString());
    }
}
