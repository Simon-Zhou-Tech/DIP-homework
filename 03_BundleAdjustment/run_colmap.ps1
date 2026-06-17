# COLMAP 3D reconstruction pipeline for Windows (PowerShell)
# Usage: .\run_colmap.ps1

$ErrorActionPreference = "Stop"

$COLMAP_BIN = "D:/software/colmap-x64-windows-cuda/bin/colmap.exe"
$DATASET_PATH = "data"
$IMAGE_PATH = "$DATASET_PATH/images"
$COLMAP_PATH = "$DATASET_PATH/colmap"

New-Item -ItemType Directory -Force -Path "$COLMAP_PATH/sparse" | Out-Null
New-Item -ItemType Directory -Force -Path "$COLMAP_PATH/dense" | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
& $COLMAP_BIN feature_extractor `
    --database_path "$COLMAP_PATH/database.db" `
    --image_path "$IMAGE_PATH" `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1

Write-Host "=== Step 2: Feature Matching ==="
& $COLMAP_BIN exhaustive_matcher `
    --database_path "$COLMAP_PATH/database.db"

Write-Host "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
& $COLMAP_BIN mapper `
    --database_path "$COLMAP_PATH/database.db" `
    --image_path "$IMAGE_PATH" `
    --output_path "$COLMAP_PATH/sparse"

Write-Host "=== Step 4: Image Undistortion ==="
& $COLMAP_BIN image_undistorter `
    --image_path "$IMAGE_PATH" `
    --input_path "$COLMAP_PATH/sparse/0" `
    --output_path "$COLMAP_PATH/dense"

Write-Host "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
& $COLMAP_BIN patch_match_stereo `
    --workspace_path "$COLMAP_PATH/dense"

Write-Host "=== Step 6: Stereo Fusion ==="
& $COLMAP_BIN stereo_fusion `
    --workspace_path "$COLMAP_PATH/dense" `
    --output_path "$COLMAP_PATH/dense/fused.ply"

Write-Host "=== Done! ==="
Write-Host "Results:"
Write-Host "  Sparse: $COLMAP_PATH/sparse/0/"
Write-Host "  Dense:  $COLMAP_PATH/dense/fused.ply"
