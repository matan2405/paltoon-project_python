using UnityEngine;

[RequireComponent(typeof(Camera))]
public class RearMirrorSetup : MonoBehaviour
{
    void Awake()
    {
        var cam = GetComponent<Camera>();
        var rt  = cam.targetTexture;
        if (rt == null) return;

        // Re-assign forces Unity to re-bind the RT after any asset resize
        cam.targetTexture = null;
        cam.targetTexture = rt;
    }
}
