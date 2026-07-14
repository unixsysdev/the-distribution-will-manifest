# Artifact index

`model-specs/launch/` preserves the text specifications and training notes recovered for the launch-model experiments. The original directory names are retained beneath that one archive folder so they remain traceable to the paper and chronological notebook.

These are documentary records, not deployable model bundles. Trained `.pkl` files, derived datasets, and live runtime state are intentionally absent.

The research scripts still use root-level `bot_artifacts_*` paths as their **local generated-output contract**. Those directories are deployment/runtime state, are ignored or untracked in the public release, and should not be confused with the archived specifications here.
