using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

[RequireComponent(typeof(AdvancedBicycleModel))]
public class VehicleDataVisualizer : MonoBehaviour
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
        public float Fxf;
        public float lambda;
        //Outputs
        public Vector3 position;
        public float vx;
        public float ax;
        public float y;                          // Lateral position
        public float yDot;                       // Lateral velocity
        public float psi;                        // Yaw angle
        public float psiDot;
        public float EngineRpm;

        public DataPoint(float time, float Fxf, float lambda, Vector3 position, float vx, float ax, float y, float yDot, float psi, float psiDot, float engineRpm)
        {
            this.time = time;
            this.Fxf = Fxf;
            this.lambda = lambda;
            this.position = position;
            this.vx = vx;
            this.ax = ax;
            this.y = y;
            this.yDot = yDot;
            this.psi = psi;
            this.psiDot = psiDot;
            EngineRpm = engineRpm;
        }
    }
    private List<DataPoint> dataPoints = new List<DataPoint>();
    DataPoint point;
    private float lastPlotUpdateTime;
    private LineRenderer velocityPlotRenderer;
    private LineRenderer trajectoryPlotRenderer;
    private bool isRecording = true;
    public AdvancedBicycleModel vehicle;

    private Rigidbody rb;

    private void Start()
    {
        vehicle = GetComponent<AdvancedBicycleModel>();
        rb = GetComponent<Rigidbody>();
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
        vehicle.GetFx_tot(),
        vehicle.GetLambda(),
        vehicle.transform.position,
        vehicle.GetVx(),
        vehicle.GetAx(),
        vehicle.GetY(),
        vehicle.GetYDot(),
        vehicle.GetPsi(),
        vehicle.GetPsiDot(),
        vehicle.GetEngineRpm()
    );
            dataPoints.Add(point);
            UnityEngine.Debug.Log($"Added DataPoint: {JsonUtility.ToJson(point)}");
            UnityEngine.Debug.Log($"🔍 Data points count: {dataPoints.Count}");
        }
    }
    public DataPoint GetLastDataPoint()
    {
        if (dataPoints.Count > 0)
        {
            return dataPoints.Last();
        }
        else
        {
            UnityEngine.Debug.LogWarning("No data points available.");
            return null;
        }
    }
    public void SaveDataToCSV()
    {
        string filename_in = "";
        string filename_out = $"Outputs.csv";
        UnityEngine.Debug.Log($"⚡ SaveDataToCSV() called! Data count: {dataPoints.Count}");

        if (dataPoints.Count == 0)
        {
            UnityEngine.Debug.LogWarning("No data points to save.");
            return;
        }
        if (rb.tag == "Player")
        {
            filename_in = $"Inputs_Player.csv";
            filename_out = $"Outputs_Player.csv";
        }
        else
        {
            filename_in = $"Inputs_{vehicle.GetIndexPlatoon()}.csv";
            filename_out = $"Outputs_{vehicle.GetIndexPlatoon()}.csv";
        }
        string path_in = Path.Combine(Application.dataPath, "..", "VehicleData", filename_in);
        Directory.CreateDirectory(Path.GetDirectoryName(path_in));

        string path_out = Path.Combine(Application.dataPath, "..", "VehicleData", filename_out);
        Directory.CreateDirectory(Path.GetDirectoryName(path_out));

        StringBuilder sb_in = new StringBuilder();
        sb_in.AppendLine("Time,Fx,Lambda");

        StringBuilder sb_out = new StringBuilder();
        sb_out.AppendLine("Time,PositionX,PositionY,PositionZ,Vx,Ax,Y,YDot,Psi,PsiDot,EngineRpm");

        foreach (var point in dataPoints)
        {
            sb_in.AppendLine($"{point.time},{point.Fxf},{point.lambda}");
            //UnityEngine.Debug.Log($"{point.time},{point.Fxf},{point.lambda}");
            sb_out.AppendLine($"{point.time},{point.position.x},{point.position.y},{point.position.z},{point.vx},{point.ax},{point.y},{point.yDot},{point.psi},{point.psiDot},{point.EngineRpm}");
            //UnityEngine.Debug.Log($"{point.time},{point.position.x},{point.position.y},{point.position.z},{point.vx},{point.ax},{point.y},{point.yDot},{point.psi},{point.psiDot},{point.EngineRpm}");
        }

        File.WriteAllText(path_in, sb_in.ToString());
        File.WriteAllText(path_out, sb_out.ToString());
        UnityEngine.Debug.Log($"Data saved to: {path_in} and {path_out}");
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