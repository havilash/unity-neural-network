using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.NeuralNetwork.Cost
{
    public class Cost
    {
        public readonly struct MeanSquaredError : ICost
        {
            public double Function(double[] predictedOutputs, double[] expectedOutputs)
            {
                double cost = 0;
                for (int i = 0; i < predictedOutputs.Length; i++)
                {
                    double error = predictedOutputs[i] - expectedOutputs[i];
                    cost += error * error;
                }
                return 0.5 * cost;
            }

            public double Derivative(double predictedOutput, double expectedOutput)
            {
                return predictedOutput - expectedOutput;
            }
        }

        public readonly struct CrossEntropy : ICost
        {
            // Note: expected outputs are expected to all be either 0 or 1
            public double Function(double[] predictedOutputs, double[] expectedOutputs)
            {
                double cost = 0;
                for (int i = 0; i < predictedOutputs.Length; i++)
                {
                    double x = predictedOutputs[i];
                    double y = expectedOutputs[i];
                    double v = (y == 1) ? -Math.Log(x) : -Math.Log(1 - x);
                    cost += double.IsNaN(v) ? 0 : v;
                }
                return cost;
            }

            public double Derivative(double predictedOutput, double expectedOutput)
            {
                double x = predictedOutput;
                double y = expectedOutput;
                if (x == 0 || x == 1)
                {
                    return 0;
                }
                return (-x + y) / (x * (x - 1));
            }
        }

    }
}
