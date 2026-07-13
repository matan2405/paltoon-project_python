using UnityEngine;

public class ObjectSizeChecker : MonoBehaviour
{
    void Start()
    {
        CheckSize();
    }

    [ContextMenu("Check Total Size")]
    public void CheckSize()
    {
        Bounds combinedBounds = new Bounds(transform.position, Vector3.zero);
        Renderer[] renderers = GetComponentsInChildren<Renderer>();

        if (renderers.Length == 0)
        {
            Debug.LogWarning($"No Renderer components found on {gameObject.name} or its children.");
            return;
        }

        foreach (Renderer render in renderers)
        {
            combinedBounds.Encapsulate(render.bounds);
        }

        Vector3 size = combinedBounds.size;
        Debug.Log($"<b>Object Bounds Size for {gameObject.name}:</b>\n" +
                  $"Width  (X): {size.x:F2} meters\n" +
                  $"Height (Y): {size.y:F2} meters\n" +
                  $"Length (Z): {size.z:F2} meters");
    }
}