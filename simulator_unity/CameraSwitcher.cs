using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera firstPersonCamera;
    public Camera thirdPersonCamera;

    private bool isThirdPerson = true;

    // Start is called before the first frame update
    void Start()
    {
        // הפעלת המצלמה השלישית-אישית בתחילת המשחק
        firstPersonCamera.enabled = false;
        thirdPersonCamera.enabled = true;
    }

    // Update is called once per frame
    void Update()
    {
        // בדיקה אם המשתמש לוחץ על מקש C להחלפת מצלמה
        if (Input.GetKeyDown(KeyCode.C))
        {
            isThirdPerson = !isThirdPerson;
            SwitchCamera();
        }
    }

    void SwitchCamera()
    {
        if (isThirdPerson)
        {
            firstPersonCamera.enabled = false;
            thirdPersonCamera.enabled = true;
        }
        else
        {
            firstPersonCamera.enabled = true;
            thirdPersonCamera.enabled = false;
        }
    }
}
