using System;
using System.Collections;
using System.Collections.Generic;
using System.Transactions;
using UnityEngine;


public class NeuralNetworkMain : MonoBehaviour
{
    NeuralNetwork neuralNetwork = new NeuralNetwork(1, 1);

    DataPoint[] data = new DataPoint[]
    {
        new DataPoint(new double[] { 1.0 }, new double[] { 1.0 }),
        new DataPoint(new double[] { 0.0 }, new double[] { 0.0 }),

        //new DataPoint(new double[] { 1.0, 0.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 0.0, 1.0 }, new double[] { 0.0, 1.0 }),

        //new DataPoint(new double[] { 5.0, 13.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 7.0, 8.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 2.0, 6.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 8.0, 3.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 3.0, 10.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 6.0, 7.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 1.0, 5.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 7.0, 2.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 4.0, 9.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 5.0, 6.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 0.0, 4.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 6.0, 1.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 3.0, 8.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 4.0, 5.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 5.0, 2.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 1.0, 7.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 2.0, 3.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 6.0, 0.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 3.0, 6.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 4.0, 1.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 5.0, 5.0 }, new double[] { 1.0, 0.0 }),
        //new DataPoint(new double[] { 0.0, 2.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 1.0, 6.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 2.0, 1.0 }, new double[] { 0.0, 1.0 }),
        //new DataPoint(new double[] { 6.0, 5.0 }, new double[] { 1.0, 0.0 }),
    };

    // Start is called before the first frame update
    void Start()
    {


        print(String.Join(" | ", neuralNetwork.CalculateOutputs(data[0].inputs)));
        print(String.Join(" | ", neuralNetwork.CalculateOutputs(data[1].inputs)));

        print("");
        print("");

        for (int i = 1; i < 20; i++)
        {
            foreach (DataPoint dataPoint in data)
            {
                neuralNetwork.UpdateAllGradients(dataPoint);
            }
            double[,] gradientW = neuralNetwork.layers[0].costGradientW;
            double[] gradientB = neuralNetwork.layers[0].costGradientB;

            //print("weight: " + neuralNetwork.layers[0].weights[0, 0].ToString() + " | " + neuralNetwork.layers[0].weights[0, 1].ToString());
            //print("weigth: " + neuralNetwork.layers[0].weights[1, 0].ToString() + " | " + neuralNetwork.layers[0].weights[1, 1].ToString());
            //print("gradient: " + gradient[0, 0].ToString() + " | " + gradient[0, 1].ToString());
            //print("gradient: " + gradient[1, 0].ToString() + " | " + gradient[1, 1].ToString());

            print("weigth: " + neuralNetwork.layers[0].weights[0, 0].ToString() + " | bias: " + neuralNetwork.layers[0].biases[0].ToString());
            print("gradientW: " + gradientW[0, 0].ToString() + " | gradientB: " + gradientW[0, 0].ToString());

            neuralNetwork.ClearAllGradients();

            neuralNetwork.Learn(data, .3);

            print("");
        }



        print("");

        print(String.Join(" | ", neuralNetwork.CalculateOutputs(data[0].inputs)));
        print(String.Join(" | ", neuralNetwork.CalculateOutputs(data[1].inputs)));

        //for (int i = 1; i < 500; i++)
        //{
        //    neuralNetwork.Learn(data, 0.5);
        //    print(String.Join(", ", neuralNetwork.CalculateOutputs(data[0].inputs)));
        //}

        //print("");
        //print(String.Join(", ", neuralNetwork.CalculateOutputs(new double[] { 1, 0 })));
        //print(String.Join(", ", neuralNetwork.CalculateOutputs(new double[] { 0, 1 })));
    }

    void DrawNeuralNetwork()
    {
        neuralNetwork.layers[0].weights[0, 0] = 0;

        double start = 0;
        double range = 10;
        int resolution = 30;
        for (int i = 1; i < resolution; i++)
        {
            neuralNetwork.layers[0].weights[0, 0] = i/30 * range + start;

            foreach (DataPoint dataPoint in data)
            {
                neuralNetwork.UpdateAllGradients(dataPoint);
            }
            double[,] gradient = neuralNetwork.layers[0].costGradientW;



            DrawPoint(new Vector3((float)gradient[0, 0], (float)neuralNetwork.layers[0].weights[0, 0], 0), Color.green);

            neuralNetwork.ClearAllGradients();
        }
        
    }

    private void DrawPoint(Vector3 pos, Color color)
    {
        Debug.DrawLine(pos, pos, color);
    }
}
