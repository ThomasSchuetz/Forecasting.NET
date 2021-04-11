using MathNet.Numerics.LinearAlgebra.Double;
using Xunit;

namespace LinearRegression.Test
{
    public class RegressionTests
    {
        readonly DenseMatrix _features;
        readonly DenseVector _labels;
        readonly DenseVector _coefficients;

        public RegressionTests()
        {
            // Test data taken from here (page 8): 
            // http://mezeylab.cb.bscb.cornell.edu/labmembers/documents/supplement%205%20-%20multiple%20regression.pdf

            _features = DenseMatrix.OfArray(new[,] {
                { 1.0, 7.0, 560 },
                { 1.0, 3.0, 220 },
                { 1.0, 3.0, 340 },
                { 1.0, 4.0, 80 },
                { 1.0, 6.0, 150 },
                { 1.0, 7.0, 330 },
            });

            _labels = DenseVector.OfArray(new[] 
                { 16.68, 11.5, 12.03, 14.88, 13.75, 18.11 }
            );

            // own calculations, not part of original test data
            _coefficients = DenseVector.OfArray(new[] { 8.56558, 1.19652, -0.00020 });
        }


        [Fact]
        public void RegressionReturnsCorrectModelData()
        {
            var model = new Regression();

            model.Fit(_features, _labels);

            var result = model.Coefficients;
            var expected = _coefficients;

            var difference = result.Subtract(expected);

            Assert.True(difference.AbsoluteMinimum() <= 0.001);
        }

        [Fact]
        public void R2ScoreIsAbove75PercentForTrainingData()
        {
            var model = new Regression();

            model.Fit(_features, _labels);

            Assert.True(model.Score(_features, _labels, EvaluationScores.RSquared) > 0.75);
        }

        [Fact]
        public void PredictingFromUnfittedModelThrowsException()
        {
            var model = new Regression();
            Assert.Throws<UnfittedModelException>(
                () => model.Predict(SparseMatrix.CreateIdentity(3)));
        }

        [Fact]
        public void GettingCoefficienctsFromUnfittedModelThrowsException()
        {
            var model = new Regression();
            Assert.Throws<UnfittedModelException>(() => model.Coefficients);
        }
    }
}
