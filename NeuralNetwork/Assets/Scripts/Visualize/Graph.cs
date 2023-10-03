using System;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.VisualScripting;
using UnityEngine.UI;

public class Graph : MonoBehaviour
{
    [SerializeField] private GameObject pointPrefab, linePrefab, squarePrefab;
    [SerializeField] private float xScale = 1.0f, yScale = 1.0f;
    [SerializeField] private int xTickNum = 10, yTickNum = 10;
    [SerializeField] private Color positiveColor =  new(0.25f, 0.25f, 1);
    [SerializeField] private Color negativeColor = new(1, 0.25f, 0.25f);

    private List<GameObject> points = new(), labels = new(), axes = new(), background = new();

    public void Draw(GraphPoint[] data)
    {
        RectTransform containerRectTransform = transform.parent.GetComponent<RectTransform>();
        Vector2 contianerSize = containerRectTransform.rect.size; 
        
        RectTransform graphRectTransform = GetComponent<RectTransform>();
        Vector2 graphSize = graphRectTransform.rect.size;

        ClearAll();

        foreach (GraphPoint point in data)
            DrawPoint(LerpPoint(point.position, graphSize, Vector2.zero), point.isPositive ? positiveColor : negativeColor);

        DrawLine(new Vector2(contianerSize.x, 3), new Vector2(0, -graphSize.y / 2), 0.2f);
        DrawLine(new Vector2(3, contianerSize.y), new Vector2(-graphSize.x / 2, 0), 0.2f);

        DrawGrid(contianerSize, graphSize, 0.01f);
    }

    public void DrawNN(NeuralNetwork nn, int[] gridSize)
    {
        ClearObjects(background);

        RectTransform graphRectTransform = GetComponent<RectTransform>();
        Vector2 graphSize = graphRectTransform.rect.size;

        Vector2 gridSpacing = new Vector2(xScale / gridSize[0], yScale / gridSize[1]);
        Vector2 realGridSpacing = new Vector2(graphSize.x / gridSize[0], graphSize.y / gridSize[0]);

        for (int x = 0; x < gridSize[0]; x++)
        for (int y = 0; y < gridSize[1]; y++)
            {
                Vector2 point = new Vector2(x, y) * gridSpacing + gridSpacing/2;
                Vector2 realPoint = LerpPoint(point, graphSize, Vector2.zero);

                double[] inputs = { point.x, point.y };
                bool output = !Convert.ToBoolean(nn.Classify(inputs));

                GameObject square = CreateSquare(realPoint, realGridSpacing, (output ? positiveColor : negativeColor ).WithAlpha(0.1f));
                background.Add(square);
            }
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

    private GameObject CreateObject(GameObject prefab, Vector2 position)
    {
        GameObject newObj = Instantiate(prefab, transform);
        newObj.transform.localPosition = new Vector3(position.x, position.y, 0);
        return newObj;
    }

    private GameObject CreateSquare(Vector2 center, Vector2 size, Color color)
    {
        GameObject square = CreateObject(squarePrefab, center);
        square.transform.localScale = new Vector3(size.x, size.y, 1);
        square.GetComponent<Image>().color = color;
        return square;
    }

    private void DrawPoint(Vector2 position, Color color)
    {
        GameObject newPoint = CreateObject(pointPrefab, position);
        newPoint.GetComponent<Image>().color = color;
        points.Add(newPoint);
    }

    private void DrawLine(Vector2 scale, Vector2 position, float opacity)
    {
        GameObject axis = CreateObject(linePrefab, position);

        axis.transform.localScale = new Vector3(scale.x, scale.y, 0);
        axis.GetComponent<Image>().color = new Color(1, 1, 1, opacity);

        axes.Add(axis);
    }

    private void DrawGrid(Vector2 containerSize, Vector2 graphSize, float opacity)
    {
        float xTickSize = graphSize.x / xTickNum;
        float yTickSize = graphSize.y / yTickNum;

        for (float i = 0; i < containerSize.x; i += xTickSize)
        {
            float xPos = -graphSize.x / 2 + i;
            DrawLine(new Vector2(3, containerSize.y), new Vector2(xPos, 0), opacity);
        }

        for (float i = 0; i < containerSize.y; i += yTickSize)
        {
            float yPos = -graphSize.y / 2 + i;
            DrawLine(new Vector2(containerSize.x, 3), new Vector2(0, yPos), opacity);
        }
    }

    private Vector2 LerpPoint(Vector2 point, Vector2 graphSize, Vector2 offset)
    {
        float x = Mathf.Lerp(-graphSize.x / 2, graphSize.x / 2, Math.Min(1, point.x / xScale));
        float y = Mathf.Lerp(-graphSize.y / 2, graphSize.y / 2, Math.Min(1, point.y / yScale));

        return new Vector2(x, y) - offset;
    }

    private Vector2 InverseLerpPoint(Vector2 point, Vector2 graphSize, Vector2 offset)
    {
        point += offset;

        float x = (point.x + graphSize.x / 2) * xScale;
        float y = (point.y + graphSize.y / 2) * yScale;

        return new Vector2(x, y);
    }

}
