using MathNet.Numerics.LinearAlgebra;

namespace LinearRegression
{
    public class Regression
    {
        private Vector<double> _coefficients;

        public bool IsTrained { get; private set; }
        public Vector<double> Coefficients
        {
            get
            {
                if (IsTrained)
                {
                    return _coefficients;
                }
                else
                {
                    throw new UnfittedModelException(
                        "Model has to be trained before reading model coefficients!");
                }
            }
        }

        public Regression()
        {
            IsTrained = false;
        }

        public void Fit(Matrix<double> features, Vector<double> labels)
        {
            _coefficients = MathNet.Numerics.LinearRegression.MultipleRegression.Svd(
                features, labels);
            IsTrained = true;
        }

        public double Score(
            Matrix<double> features, Vector<double> labels, EvaluationScores score)
        {
            var evaluation = EvaluateScoreFactory.Create(score);
            return evaluation.Evaluate(Predict(features), labels);
        }

        public Vector<double> Predict(Matrix<double> features)
        {
            if (!IsTrained)
            {
                throw new UnfittedModelException(
                    "Model has to be fitted before making predictions!");
            }

            return features.Multiply(Coefficients);
        }
    }
}