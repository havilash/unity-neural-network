using System;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.VisualScripting;
using UnityEngine.UI;
using System.Linq;

public class Graph : MonoBehaviour
{
    private const float STEP_SIZE = 0.01f;
    private const float LINE_WIDTH = 0.1f;
    private const double LOW_INITIAL = -10;
    private const double HIGH_INITIAL = 10;

    [SerializeField] private GameObject pointPrefab, linePrefab, squarePrefab;
    [SerializeField] private float xScale = 1.0f, yScale = 1.0f;
    [SerializeField] private int xTickNum = 10, yTickNum = 10;
    [SerializeField] private Color positiveColor = new(0.25f, 0.25f, 1);
    [SerializeField] private Color negativeColor = new(1, 0.25f, 0.25f);

    private List<GameObject> points = new(), labels = new(), axes = new(), background = new();

    public void Draw(GraphPoint[] data)
    {
        RectTransform containerRectTransform = transform.parent.GetComponent<RectTransform>();
        Vector2 containerSize = containerRectTransform.rect.size;

        RectTransform graphRectTransform = GetComponent<RectTransform>();
        Vector2 graphSize = graphRectTransform.rect.size;

        ClearAll();

        foreach (GraphPoint point in data)
            DrawPoint(LerpPoint(point.position, graphSize, Vector2.zero), point.isPositive ? positiveColor : negativeColor);

        DrawLine(new Vector2(containerSize.x, LINE_WIDTH), new Vector2(0, -graphSize.y / 2), Color.white, axes);
        DrawLine(new Vector2(LINE_WIDTH, containerSize.y), new Vector2(-graphSize.x / 2, 0), Color.white, axes);

        DrawGrid(containerSize, graphSize, STEP_SIZE);
    }

    public void DrawNN(NeuralNetwork nn, int[] gridSize, GraphPoint[] data)
    {
        ClearObjects(background);

        RectTransform graphRectTransform = GetComponent<RectTransform>();
        Vector2 graphSize = graphRectTransform.rect.size;

        Vector2 gridSpacing = new Vector2(xScale / gridSize[0], yScale / gridSize[1]);
        Vector2 realGridSpacing = new Vector2(graphSize.x / gridSize[0], graphSize.y / gridSize[0]);

        for (int x = 0; x < gridSize[0]; x++)
        for (int y = 0; y < gridSize[1]; y++)
        {
            Vector2 point = new Vector2(x, y) * gridSpacing + gridSpacing / 2;
            Vector2 realPoint = LerpPoint(point, graphSize, Vector2.zero);

            double[] inputs = { point.x, point.y };
            bool output = nn.Classify(inputs) == 0;

            GameObject square = CreateSquare(realPoint, realGridSpacing, (output ? positiveColor : negativeColor).WithAlpha(0.1f));
            background.Add(square);
        }
    }

    //public void DrawNN(NeuralNetwork neuralNetwork, int[] gridSize, GraphPoint[] data)
    //{
    //    ClearObjects(background);

    //    RectTransform graphRectTransform = GetComponent<RectTransform>();
    //    Vector2 graphSize = graphRectTransform.rect.size;

    //    Vector2 gridSpacing = new Vector2(xScale / gridSize[0], yScale / gridSize[1]);
    //    Vector2 realGridSpacing = new Vector2(graphSize.x / gridSize[0], graphSize.y / gridSize[0]);

    //    List<double[]> linePoints = GetDividingLinePoints(neuralNetwork, STEP_SIZE, data);

    //    DrawDividingLine(linePoints, graphSize);
    //    ColorAreas(neuralNetwork, gridSize, linePoints, gridSpacing, realGridSpacing, graphSize);
    //}

    private void DrawDividingLine(List<double[]> linePoints, Vector2 graphSize)
    {
        for (int i = 0; i < linePoints.Count - 1; i++)
        {
            Vector2 start = LerpPoint(new Vector2((float)linePoints[i][0], (float)linePoints[i][1]), graphSize, Vector2.zero);
            Vector2 end = LerpPoint(new Vector2((float)linePoints[i + 1][0], (float)linePoints[i + 1][1]), graphSize, Vector2.zero);

            DrawLine(start, end, Color.black, background);
        }
    }

    private void ColorAreas(NeuralNetwork neuralNetwork,
                            int[] gridSize,
                            List<double[]> linePoints,
                            Vector2 gridSpacing,
                            Vector2 realGridSpacing,
                            Vector2 graphSize)
    {
        for (int x = 0; x < gridSize[0]; x++)
            for (int y = 0; y < gridSize[1]; y++)
            {
                Vector2 point = new Vector2(x, y) * gridSpacing + gridSpacing / 2;
                Vector2 realPoint = LerpPoint(point, graphSize, Vector2.zero);

                double[] inputs = { point.x, point.y };
                bool outputIsPositive =
                    neuralNetwork.Classify(inputs) == 0;

                int indexForXValue =
                    GetIndexForXValue(point.x);
                if (indexForXValue >= linePoints.Count)
                {
                    indexForXValue =
                        linePoints.Count - 1;
                }
                bool isAboveLine =
                    point.y > linePoints[indexForXValue][1];

                Color colorForArea =
                    GetColorForArea(outputIsPositive,
                                    isAboveLine);

                GameObject square =
                    CreateSquare(realPoint,
                                 realGridSpacing,
                                 colorForArea.WithAlpha(0.1f));
                background.Add(square);
            }
    }

    public List<double[]> GetDividingLinePoints(NeuralNetwork neuralNetwork,
                                                double stepSize,
                                                GraphPoint[] data)
    {
        double xStartValueInData =
            data.Min(graphPoint => graphPoint.position.x);
        double xEndValueInData =
            data.Max(graphPoint => graphPoint.position.x);

        List<double[]> linePoints = new List<double[]>();

        for (double x = xStartValueInData;
             x <= xEndValueInData;
             x += stepSize)
        {
            double low = LOW_INITIAL;
            double high = HIGH_INITIAL;
            double mid = 0;

            while (high - low > STEP_SIZE)
            {
                mid = (low + high) / 2;
                double[] input = { x, mid };
                bool output =
                    neuralNetwork.Classify(input) == 0;

                if (output)
                {
                    low = mid;
                }
                else
                {
                    high = mid;
                }
            }

            linePoints.Add(new double[] { x, mid });
        }

        return linePoints;
    }

    private void ClearAll()
    {
        ClearObjects(points);
        ClearObjects(labels);
        ClearObjects(axes);
    }

    private void ClearObjects(List<GameObject> objects)
    {
        foreach (GameObject obj in objects)
            Destroy(obj);

        objects.Clear();
    }

    private GameObject CreateObject(GameObject prefab,
                                    Vector2 position)
    {
        GameObject newObj =
            Instantiate(prefab, transform);
        newObj.transform.localPosition =
            new Vector3(position.x, position.y, 0);
        return newObj;
    }

    private GameObject CreateSquare(Vector2 center,
                                    Vector2 size,
                                    Color color)
    {
        GameObject square =
            CreateObject(squarePrefab, center);
        square.transform.localScale =
            new Vector3(size.x, size.y, 1);
        square.GetComponent<Image>().color = color;
        return square;
    }

    private void DrawPoint(Vector2 position,
                           Color color)
    {
        GameObject newPoint =
            CreateObject(pointPrefab, position);
        newPoint.GetComponent<Image>().color = color;
        points.Add(newPoint);
    }

    public GameObject DrawLine(Vector2 start, Vector2 end, Color color, List<GameObject> container)
    {
        GameObject line = new GameObject();
        line.transform.SetParent(this.transform);  // Set the parent

        LineRenderer lineRenderer = line.AddComponent<LineRenderer>();

        lineRenderer.startColor = color;
        lineRenderer.endColor = color;

        lineRenderer.startWidth = LINE_WIDTH;
        lineRenderer.endWidth = LINE_WIDTH;

        lineRenderer.SetPosition(0, start);
        lineRenderer.SetPosition(1, end);

        container.Add(line);

        return line;
    }


    private void DrawGrid(Vector2 containerSize,
                          Vector2 graphSize,
                          float opacity)
    {
        float xTickSize = graphSize.x / xTickNum;
        float yTickSize = graphSize.y / yTickNum;

        for (float i = 0; i < containerSize.x; i += xTickSize)
        {
            float xPos = -graphSize.x / 2 + i;
            DrawLine(new Vector2(LINE_WIDTH, containerSize.y),
                     new Vector2(xPos, 0),
                     Color.white.WithAlpha(opacity),
                     axes);
        }

        for (float i = 0; i < containerSize.y; i += yTickSize)
        {
            float yPos = -graphSize.y / 2 + i;
            DrawLine(new Vector2(containerSize.x, LINE_WIDTH),
                     new Vector2(0, yPos),
                     Color.white.WithAlpha(opacity),
                     axes);
        }
    }

    private Vector2 LerpPoint(Vector2 point,
                              Vector2 graphSize,
                              Vector2 offset)
    {
        float x =
            Mathf.Lerp(-graphSize.x / 2,
                       graphSize.x / 2,
                       Math.Min(1, point.x / xScale));
        float y =
            Mathf.Lerp(-graphSize.y / 2,
                       graphSize.y / 2,
                       Math.Min(1, point.y / yScale));

        return new Vector2(x, y) - offset;
    }

    private int GetIndexForXValue(float x)
    {
        return (int)(x / STEP_SIZE);
    }

    private Color GetColorForArea(bool outputIsPositive,
                                  bool isAboveLine)
    {
        if (outputIsPositive)
            return isAboveLine ? positiveColor : negativeColor;
        else
            return isAboveLine ? negativeColor : positiveColor;
    }
}
