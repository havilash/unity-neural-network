using System.Linq;
using Assets.Scripts.NeuralNetwork.Activation;
using Assets.Scripts.NeuralNetwork.Cost;

public class NeuralNetwork
{
    public Layer[] layers;
    public readonly int[] layerSizes;

    public ICost cost;

    System.Random rng;

    public NeuralNetwork(int[] layerSizes)
    {
        this.layerSizes = layerSizes;
        this.cost = new Cost.MeanSquaredError();
        rng = new System.Random();

        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    public void SetCost(ICost costFunction)
    {
        this.cost = costFunction;
    }

    public void SetActivation(IActivation activation)
    {
        SetActivation(activation, activation);
    }

    public void SetActivation(IActivation activation, IActivation outputLayerActivation)
    {
        for (int i = 0; i < layers.Length - 1; i++)
        {
            layers[i].SetActivationFunction(activation);
        }
        layers[layers.Length - 1].SetActivationFunction(outputLayerActivation);
    }

    public double[] CalculateOutputs(double[] inputs)
    {
        foreach (Layer layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    public int Classify(double[] inputs)
    {
        double[] outputs = CalculateOutputs(inputs);
        return outputs.ToList().IndexOf(outputs.Max());
    }

    double Cost(DataPoint dataPoint)
    {
        double[] outputs = CalculateOutputs(dataPoint.inputs);
        double cost = this.cost.Function(outputs, dataPoint.expectedOutputs);

        return cost;
    }

    double AvgCost(DataPoint[] data)
    {
        double totalCost = 0;

        foreach (DataPoint dataPoint in data)
        {
            totalCost += Cost(dataPoint);
        }

        return totalCost / data.Length;
    }

    public void Learn(DataPoint[] trainingBatch, double learnRate)
    {
        foreach(DataPoint dataPoint in trainingBatch)
        {
            UpdateAllGradients(dataPoint);
        }

        ApplyAllGradients(learnRate / trainingBatch.Length);

        ClearAllGradients();
    }

    public void ClearAllGradients()
    {
        foreach (Layer layer in layers)
        {
            layer.ClearGradients();
        }
    }

    public void ApplyAllGradients(double learnRate)
    {
        foreach (Layer layer in layers)
        {
            layer.ApplyGradients(learnRate);
        }
    }

    public void UpdateAllGradients(DataPoint dataPoint)
    {
        CalculateOutputs(dataPoint.inputs);

        Layer outputLayer = layers[layers.Length - 1];
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs, cost);
        outputLayer.UpdateGradients(nodeValues);

        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) 
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
        
    }

}
