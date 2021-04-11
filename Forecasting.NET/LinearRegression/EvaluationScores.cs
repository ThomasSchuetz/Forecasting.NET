using System;
using System.Collections.Generic;

namespace LinearRegression
{
    public enum EvaluationScores
    {
        CoefficientOfDetermination,
        PopulationStandardError,
        R,
        RSquared
    }

    public interface IEvaluateScore
    {
        double Evaluate(IEnumerable<double> modelledValues, IEnumerable<double> observedValues);
    }

    public class CoefficientOfDetermination : IEvaluateScore
    {
        public double Evaluate(IEnumerable<double> modelledValues, IEnumerable<double> observedValues)
            => MathNet.Numerics.GoodnessOfFit.CoefficientOfDetermination(modelledValues, observedValues);
    }

    public class PopulationStandardError : IEvaluateScore
    {
        public double Evaluate(IEnumerable<double> modelledValues, IEnumerable<double> observedValues)
            => MathNet.Numerics.GoodnessOfFit.PopulationStandardError(modelledValues, observedValues);
    }

    public class R : IEvaluateScore
    {
        public double Evaluate(IEnumerable<double> modelledValues, IEnumerable<double> observedValues)
            => MathNet.Numerics.GoodnessOfFit.R(modelledValues, observedValues);
    }

    public class RSquared : IEvaluateScore
    {
        public double Evaluate(IEnumerable<double> modelledValues, IEnumerable<double> observedValues)
            => MathNet.Numerics.GoodnessOfFit.RSquared(modelledValues, observedValues);
    }

    public static class EvaluateScoreFactory
    {
        public static IEvaluateScore Create(EvaluationScores score)
        {
            return score switch
            {
                EvaluationScores.CoefficientOfDetermination => new CoefficientOfDetermination(),
                EvaluationScores.PopulationStandardError => new PopulationStandardError(),
                EvaluationScores.R => new R(),
                EvaluationScores.RSquared => new RSquared(),
                _ => throw new ArgumentException($"Undefined evaluation score ({score})"),
            };
        }
    }
}
