using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor.UIElements;
using UnityEngine.XR;

public class NeuralNetwork
{
    public Layer[] layers;
    public readonly int[] layerSizes;

    Random rng;

    public NeuralNetwork(params int[] layerSizes)
    {
        this.layerSizes = layerSizes;
        rng = new Random();

        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }
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
        Layer outputLayer = layers[layers.Length - 1];
        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.Length; nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
        }

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
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
        outputLayer.UpdateGradients(nodeValues);

        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) 
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
        
    }

}
