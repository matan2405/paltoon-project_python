using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController1 : MonoBehaviour
{
    public Transform player;
    public Transform driver;
    private Rigidbody playerRB;
    public Vector3 Offset;
    public AdvancedBicycleModel vehicle;

    public float speed;

    // Start is called before the first frame update
    void Start()
    {
        playerRB = player.GetComponent<Rigidbody>();
        speed = vehicle.GetVx();
        driver = player.Find("driver").transform;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        transform.position = driver.position;
        //transform.position = Vector3.Lerp(transform.position, driver.position + driver.TransformVector(Offset), Time.deltaTime * speed);
        transform.rotation = driver.rotation;
        //transform.LookAt(driver); // the camra will look at the driver
    }
}
