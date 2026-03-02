//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;

//public class CameraController : MonoBehaviour
//{
//    public Transform player;
//    public Transform driver;
//    private Rigidbody playerRB;
//    public Vector3 Offset;
//    public float speed;
//    // Start is called before the first frame update
//    void Start()
//    {
//        playerRB = player.GetComponent<Rigidbody>();
//        driver = player.Find("driver").transform;
//    }

//    // Update is called once per frame
//    void FixedUpdate()
//    {
//        transform.position = driver.position + driver.TransformVector(Offset);
//        transform.LookAt(driver); // המצלמה תסתכל על הנהג

//        //Vector3 playerForward = (playerRB.linearVelocity + player.transform.forward).normalized;
//        //transform.position = Vector3.Lerp(transform.position,
//        //    playerRB.position + playerRB.transform.TransformVector(Offset)
//        //    + playerForward * (-5f),
//        //    speed * Time.deltaTime);
//        //transform.LookAt(playerRB);
//    }
//}
