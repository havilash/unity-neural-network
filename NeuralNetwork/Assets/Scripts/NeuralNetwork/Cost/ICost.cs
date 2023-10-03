using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.NeuralNetwork.Cost
{
    public interface ICost
    {
        double Function(double[] predictedOutputs, double[] expectedOutputs);

        double Derivative(double predictedOutput, double expectedOutput);
    }
}
