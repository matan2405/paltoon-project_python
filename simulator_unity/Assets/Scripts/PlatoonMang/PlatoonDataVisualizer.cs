using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

[RequireComponent(typeof(PlatoonManager))]
public class PlatoonDataVisualizer : MonoBehaviour
{
    //[System.Serializable]
    //public class PlotSettings
    //{
    //    public bool showPlots = true;
    //    public float plotWidth = 400f;
    //    public float plotHeight = 300f;
    //    public Color velocityPlotColor = Color.blue;
    //    public Color trajectoryPlotColor = Color.green;
    //    public float plotUpdateInterval = 0.1f;
    //    public int maxDataPoints = 1000;
    //}

    [System.Serializable]
    public class DataPoint
    {
        public float time;
        //Inputs
        public float actual_gap1;
        public float actual_gap2;
        //Outputs
        public Vector3 position;
        public float des_gap1;
        public float des_gap2;
        public float TargetVelocity;
        public DataPoint(float time, float actual_gap1, float actual_gap2, float des_gap1, float des_gap2, float TargetVelocity)
        {
            this.time = time;
            this.actual_gap1 = actual_gap1;
            this.actual_gap2 = actual_gap2;
            this.des_gap1 = des_gap1;
            this.des_gap2 = des_gap2;
            this.TargetVelocity = TargetVelocity;
        }
    }
    private List<DataPoint> dataPoints = new List<DataPoint>();
    DataPoint point;
    private float lastPlotUpdateTime;
    private LineRenderer velocityPlotRenderer;
    private LineRenderer trajectoryPlotRenderer;
    private bool isRecording = true;
    private PlatoonManager platoonManager;

    private void Start()
    {
        platoonManager = GetComponent<PlatoonManager>();
        UnityEngine.Debug.Log("FixedUpdate");
    }

    private void FixedUpdate()
    {
        UnityEngine.Debug.Log($"FixedUpdate is running at {Time.fixedTime}");

        if (isRecording)
        {
            UnityEngine.Debug.Log($"Recording data at time: {Time.fixedTime}");
            // Update data points
            DataPoint point = new DataPoint(
            Time.fixedTime,
            platoonManager.GetActualGap(1), platoonManager.GetActualGap(2),
            platoonManager.GetDesiredGap(1), platoonManager.GetDesiredGap(2), platoonManager.GetTargetVelocity()
);
            dataPoints.Add(point);

            UnityEngine.Debug.Log($"Added DataPoint: {JsonUtility.ToJson(point)}");
            UnityEngine.Debug.Log($"🔍 Data points count: {dataPoints.Count}");
            UnityEngine.Debug.Log($"time: {dataPoints.Last().time}, actual_gap1: {dataPoints.Last().actual_gap1}, actual_gap2: {dataPoints.Last().actual_gap2}, des_gap1: {dataPoints.Last().des_gap1}, des_gap2: {dataPoints.Last().des_gap2}, TargetVelocity: {dataPoints.Last().TargetVelocity}");
        }
    }
    public void SaveDataToCSV()
    {
        string filename_platoon = "platoon_data.csv";
        UnityEngine.Debug.Log($"⚡ SaveDataToCSV() called! Data count: {dataPoints.Count}");


        string path_platoon = Path.Combine(Application.dataPath, "..", "VehicleData", filename_platoon);
        Directory.CreateDirectory(Path.GetDirectoryName(path_platoon));

        StringBuilder sb_platoon = new StringBuilder();
        sb_platoon.AppendLine("Time,actual_gap1,actual_gap2,des_gap1,des_gap2,TargetVelocity");

        foreach (var point in dataPoints)
        {
            sb_platoon.AppendLine($"{point.time},{point.actual_gap1},{point.actual_gap2},{point.des_gap1},{point.des_gap2},{point.TargetVelocity}");
        }

        File.WriteAllText(path_platoon, sb_platoon.ToString());
        UnityEngine.Debug.Log($"Data saved to: {path_platoon}");
    }

    private void OnDestroy()
    {
        SaveDataToCSV();
        //string exePath = @"C:\Users\matan\OneDrive - post.bgu.ac.il\מסמכים\תואר שני\car simulator with git\car_simulator_unity1\VehicleData\BicycleModelStateSpace_ode45.exe";
        //try
        //{
        //    Process.Start(exePath);
        //    UnityEngine.Debug.Log("EXE started successfully!");
        //}
        //catch (System.Exception e)
        //{
        //    UnityEngine.Debug.LogError("Failed to start EXE: " + e.Message);
        //}
    }
}