using System;
using System.Collections;
using System.Collections.Generic;
using System.Transactions;
using Assets.Scripts.NeuralNetwork.Activation;
using Assets.Scripts.NeuralNetwork.Cost;
using UnityEngine;

public class Layer
{
    public int numNodesIn, numNodesOut;

    public double[,] costGradientW;
    public double[] costGradientB;

    public double[,] weights;
    public double[] biases;

    public IActivation activation;

    double[] inputs;
    double[] weightedInputs;
    double[] activations;

    public Layer(int numNodesIn, int numNodesOut)
    {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        this.activation = new Activation.ReLU();

        weights = new double[numNodesIn, numNodesOut];
        biases = new double[numNodesOut];

        costGradientW = new double[numNodesIn, numNodesOut];
        costGradientB = new double[numNodesOut];

        InitializeRandomWeights();
    }

    // Gradient Descent
    public void ApplyGradients(double learnRate)
    {
        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientB[nodeOut] * learnRate;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn, nodeOut] -= costGradientW[nodeIn, nodeOut] * learnRate;
            }
        }
    }

    public void ClearGradients()
    {

        Array.Clear(costGradientW, 0, costGradientW.Length);
        Array.Clear(costGradientB, 0, costGradientB.Length);
    }

    public double[] CalculateOutputs(double[] inputs)
    {
        this.inputs = inputs;
        weightedInputs = new double[numNodesOut];
        activations = new double[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) 
        {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) 
            {
                weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut];
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            activations[nodeOut] = activation.Function(weightedInputs, nodeOut);
        }

        return activations;
    }

    public double NodeCostDerivative(double outputActivation, double expectedOutputs)
    {
        return 2 * (outputActivation - expectedOutputs);
    }

    public void InitializeRandomWeights()
    {
        System.Random rng = new System.Random();

        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                double randomValue = rng.NextDouble() * 2 - 1;
                weights[nodeIn, nodeOut] = randomValue / Math.Sqrt(numNodesIn);

            }
    }

    public double[] CalculateOutputLayerNodeValues(double[] expectedOutputs, ICost cost)
    {
        double[] nodeValues = new double[expectedOutputs.Length];

        for (int i = 0; i < nodeValues.Length; i++)
        {
            double costDerivative = cost.Derivative(activations[i], expectedOutputs[i]);
            double activationDerivative = activation.Derivative(weightedInputs, i);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        return nodeValues;
    }

    public double[] CalculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues)
    {
        double[] newNodeValues = new double[numNodesOut];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.Length; newNodeIndex++)
        {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
            {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldLayer.weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= activation.Derivative(weightedInputs, newNodeIndex);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }


    public void UpdateGradients(double[] nodeValues)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                double derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];
                costGradientW[nodeIn, nodeOut] += derivativeCostWrtWeight;
            }

            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }

    internal void SetActivationFunction(IActivation activation)
    {
        this.activation = activation;
    }
}
