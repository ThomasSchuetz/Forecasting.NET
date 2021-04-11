using System;

namespace LinearRegression
{
    public class UnfittedModelException : InvalidOperationException
    {
        public UnfittedModelException(string message) : base(message)
        {
        }
    }
}
