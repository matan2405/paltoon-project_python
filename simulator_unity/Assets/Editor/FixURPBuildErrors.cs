using System;
using System.Reflection;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

// Workaround for Unity 6 / URP 17.3 bug:
// "Host type is not matching any asset type" for AutodeskInteractive.shadergraph
// and TraceVirtualOffset.urtshader blocks the build.
//
// Root cause: SetRenderPipelineGlobalSettingsAsset triggers PopulateRenderPipelineGraphicsSettings
// which calls TryReloadContainedNullFields. This tries to load .shadergraph and .urtshader
// files but their ScriptedImporters (ShaderGraphImporter, URTShaderImporter) are not yet
// registered in the AppDomain at build-preprocess time — so Unity logs "Host type not matching"
// as an Error (not a warning), which counts toward the "4 errors" that fail the build.
//
// Fix: force-load the assemblies that register those importers before URP's preprocessor runs.
[InitializeOnLoad]
public class FixURPBuildErrors : IPreprocessBuildWithReport
{
    // Must be lower (earlier) than URPPreprocessBuild which is int.MinValue + 100
    public int callbackOrder => int.MinValue;

    static FixURPBuildErrors()
    {
        EditorApplication.delayCall += ForceLoadImporterAssemblies;
    }

    static void ForceLoadImporterAssemblies()
    {
        // Force ShaderGraph assembly to load so ShaderGraphImporter gets registered
        ForceAssembly("Unity.ShaderGraph.Editor");
        // Force Core RP editor assembly for URTShaderImporter
        ForceAssembly("Unity.RenderPipelines.Core.Editor");
        ForceAssembly("Unity.RenderPipelines.Universal.Editor");
    }

    static void ForceAssembly(string assemblyName)
    {
        try
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (asm.GetName().Name == assemblyName)
                    return; // already loaded
            }
            Assembly.Load(assemblyName);
        }
        catch { /* ignore — assembly may not exist or already loaded */ }
    }

    public void OnPreprocessBuild(BuildReport report)
    {
        ForceLoadImporterAssemblies();

        // Give Unity a moment to register the scripted importers
        AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
    }
}
