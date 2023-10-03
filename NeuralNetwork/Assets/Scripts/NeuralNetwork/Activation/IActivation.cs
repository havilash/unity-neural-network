using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.NeuralNetwork.Activation
{
    public interface IActivation
    {
        double Function(double[] inputs, int index);

        double Derivative(double[] inputs, int index);
    }
}
