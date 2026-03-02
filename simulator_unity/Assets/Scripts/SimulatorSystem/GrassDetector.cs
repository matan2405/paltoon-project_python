using UnityEngine;
using System.Runtime.InteropServices;
using TMPro;

public class OutMerge_or_Finish_Message : MonoBehaviour
{
    [Header("Wheel Settings")]
    public Transform[] wheelPositions;
    public float rayDistance = 1f;
    public TMP_Text messageText;
    private bool gameEnded = false;



    void Update()
    {
        if (!gameEnded)
        {
            CheckWheelsOnGrass();
        }
    }
    void EndRace(string reason)
    {
        string message = "";

        switch (reason)
        {
            case "OFF_ROAD":
                message = $"OFF ROAD!\nTwo wheels on grass\nSimulation ending...";
                Debug.Log($"🌱 Race ended - Off road");
                break;

            case "FINISH":
                message = $"FINISH!\nSimulation completed successfully!\nSimulation ending...";
                Debug.Log($"🏁 Simulation completed!");
                break;
        }
        if (messageText != null)
        {
            messageText.text = message;
            StartCoroutine(QuitAfterDelay());
            gameEnded = true;
        }
        else
        {
            Debug.LogWarning("Message Text is not assigned in the inspector.");
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("finish line") && !gameEnded)
        {
            gameEnded = true;
            EndRace("FINISH");
        }
    }
    void CheckWheelsOnGrass()
    {
        int wheelsOnGrass = 0;

        foreach (Transform wheel in wheelPositions)
        {
            RaycastHit hit;
            if (Physics.Raycast(wheel.position, Vector3.down, out hit, rayDistance))
            {
                if (hit.collider.CompareTag("Grass"))
                {
                    wheelsOnGrass++;
                }
            }
        }

        if (wheelsOnGrass >= 2)
        {
            gameEnded = true;
            EndRace("OFF_ROAD");
        }
    }
    System.Collections.IEnumerator QuitAfterDelay()
    {
        yield return new WaitForSeconds(3f);
        
        #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
        #else
            Application.Quit();
        #endif
    }
}