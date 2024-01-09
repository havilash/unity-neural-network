using Assets.Scripts.DataHandling;
using System;
using System.Collections.Generic;
using Assets.Scripts.NeuralNetwork.Activation;
using UnityEngine;
using Assets.Scripts.NeuralNetwork.Cost;

public class NeuralNetworkMain : MonoBehaviour
{
    NeuralNetwork neuralNetwork;
    [SerializeField] private Graph graph;

    DataPoint[] data;
    GraphPoint[] graphData;

    int i = 0;

    void Start()
    {
        neuralNetwork = CreateModel();
        data = LoadData("Assets\\Data\\Fruit\\Fruit_Dataset.csv");
        graphData = ConvertToGraphPoints(data);
        graph.Draw(graphData);
    }

    void Update()
    {
        if (i < 10)
        {
            neuralNetwork.Learn(data, 0.3);
            i++;
        }
        else
        {
            graph.DrawNN(neuralNetwork, new[] { 50, 50 }, graphData);
            i = 0;

            //foreach (var item in data)
            //{
            //    var inputs = String.Join(", ", neuralNetwork.CalculateOutputs(item.inputs));
            //    var expectedOutputs = String.Join(", ", item.expectedOutputs);
            //    print($"{inputs} | {expectedOutputs}");
            //}

        }
    }

    NeuralNetwork CreateModel()
    {
        NeuralNetwork nn = new NeuralNetwork(new []{2, 5, 2});
        nn.SetCost(new Cost.MeanSquaredError());
        nn.SetActivation(new Activation.ReLU());
        nn.layers[^1].SetActivation(new Activation.Softmax());
        return nn;
    }

    DataPoint[] LoadData(string path)
    {
        List<Dictionary<string, string>> rawData = DataLoader.ReadCSV(path);
        DataPoint[] data = new DataPoint[rawData.Count];

        for (int i = 0; i < data.Length; i++)
        {
            double[] inputs = { Convert.ToDouble(rawData[i]["spike_length"]) / 10, Convert.ToDouble(rawData[i]["size"]) / 1000 };
            double[] expectedOutputs = DataPoint.CreateOneHot(Convert.ToInt16(rawData[i]["is_poisonous"]), 2);
            DataPoint datapoint = new(inputs, expectedOutputs);
            data[i] = datapoint;
        }

        return data;
    }

    GraphPoint[] ConvertToGraphPoints(DataPoint[] data)
    {
        GraphPoint[] graphData = new GraphPoint[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            DataPoint dp = data[i];
            Vector2 position = new Vector2((float)dp.inputs[0], (float)dp.inputs[1]);
            bool isPositive = dp.expectedOutputs[0] == 1;
            graphData[i] = new GraphPoint(position, isPositive);
        }

        return graphData;
    }
}