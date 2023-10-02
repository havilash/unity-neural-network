using Assets.Scripts.DataHandling;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Transactions;
using UnityEngine;

public class NeuralNetworkMain : MonoBehaviour
{
    NeuralNetwork neuralNetwork = new(2, 8, 2);
    [SerializeField] private Graph graph;

    DataPoint[] data;

    int i = 0;

    void Start()
    {
        List<Dictionary<string, string>> rawData = DataLoader.ReadCSV("Assets\\Data\\Fruit\\Fruit_Dataset.csv");
        data = new DataPoint[rawData.Count];

        for (int i = 0; i < data.Length; i++)
        {
            double[] inputs = { Convert.ToDouble(rawData[i]["spike_length"])/10, Convert.ToDouble(rawData[i]["size"])/1000 };
            double[] expectedOutputs = DataPoint.CreateOneHot(Convert.ToInt16(rawData[i]["is_poisonous"]), 2);
            DataPoint datapoint = new(inputs, expectedOutputs);
            data[i] = datapoint;
        }

        GraphPoint[] graphData = new GraphPoint[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            DataPoint dp = data[i];
            Vector2 point = new Vector2((float)dp.inputs[0], (float)dp.inputs[1]);
            bool color = dp.expectedOutputs[0] == 1;
            graphData[i] = new GraphPoint(point, color);
        }

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
            graph.DrawNN(neuralNetwork, new[] { 50, 50 });
            i = 0;
        }
    }
}
