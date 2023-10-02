public struct DataPoint
{
    public readonly double[] inputs;
    public readonly double[] expectedOutputs;

    public DataPoint(double[] inputs, double[] expectedOutputs)
    {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    public static double[] CreateOneHot(int index, int num)
    {
        double[] oneHot = new double[num];
        oneHot[index] = 1;
        return oneHot;
    }

    public override string ToString()
    {
        return string.Format("DataPoint: Inputs({0}), Expected Outputs({1})",
            string.Join(", ", this.inputs),
            string.Join(", ", this.expectedOutputs));
    }
}