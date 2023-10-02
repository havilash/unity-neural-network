using UnityEngine;

public struct GraphPoint
{
    public readonly Vector2 position;
    public readonly bool isPositive;

    public GraphPoint(Vector2 position, bool isPositive)
    {
        this.position = position;
        this.isPositive = isPositive;
    }

    public override string ToString()
    {
        return string.Format("DataPoint: Inputs({0}), Expected Outputs({1})",
            this.position.ToString(),
            this.isPositive.ToString());
    }
}