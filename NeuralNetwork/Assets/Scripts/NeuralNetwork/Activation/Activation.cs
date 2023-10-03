using static System.Math;

namespace Assets.Scripts.NeuralNetwork.Activation
{
    public static class Activation
    {

        public readonly struct Sigmoid : IActivation
        {
            public double Function(double[] inputs, int index)
            {
                return 1.0 / (1 + Exp(-inputs[index]));
            }

            public double Derivative(double[] inputs, int index)
            {
                double a = Function(inputs, index);
                return a * (1 - a);
            }
        }

        public readonly struct TanH : IActivation
        {
            public double Function(double[] inputs, int index)
            {
                double e2 = Exp(2 * inputs[index]);
                return (e2 - 1) / (e2 + 1);
            }

            public double Derivative(double[] inputs, int index)
            {
                double e2 = Exp(2 * inputs[index]);
                double t = (e2 - 1) / (e2 + 1);
                return 1 - t * t;
            }
        }


        public readonly struct ReLU : IActivation
        {
            public double Function(double[] inputs, int index)
            {
                return Max(0, inputs[index]);
            }

            public double Derivative(double[] inputs, int index)
            {
                return (inputs[index] > 0) ? 1 : 0;
            }
        }

        public readonly struct SiLU : IActivation
        {
            public double Function(double[] inputs, int index)
            {
                return inputs[index] / (1 + Exp(-inputs[index]));
            }

            public double Derivative(double[] inputs, int index)
            {
                double sig = 1 / (1 + Exp(-inputs[index]));
                return inputs[index] * sig * (1 - sig) + sig;
            }
        }


        public readonly struct Softmax : IActivation
        {
            public double Function(double[] inputs, int index)
            {
                double expSum = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    expSum += Exp(inputs[i]);
                }

                double res = Exp(inputs[index]) / expSum;

                return res;
            }

            public double Derivative(double[] inputs, int index)
            {
                double expSum = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    expSum += Exp(inputs[i]);
                }

                double ex = Exp(inputs[index]);

                return (ex * expSum - ex * ex) / (expSum * expSum);
            }
        }

    }
}