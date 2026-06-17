import os
import subprocess
import argparse
import sys

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Try to find COLMAP executable
def find_colmap():
    """Find COLMAP executable, checking common locations"""
    # Check environment variable first
    colmap_exe = os.environ.get('COLMAP_EXE', '')
    if colmap_exe and os.path.exists(colmap_exe):
        return colmap_exe
    # Check common paths
    for path in [
        'colmap',  # In PATH
        r'D:\software\colmap-x64-windows-cuda\bin\colmap.exe',
        r'D:\software\colmap-x64-windows-cuda (2)\bin\colmap.exe',
        r'C:\Program Files\COLMAP\bin\colmap.exe',
        '/usr/local/bin/colmap',
    ]:
        if os.path.exists(path) or (sys.platform != 'win32' and path == 'colmap'):
            try:
                subprocess.run([path, '--help'], capture_output=True, timeout=10)
                return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    return 'colmap'  # fallback


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    parser.add_argument('--colmap_exe', type=str, default=None, help='Full path to COLMAP executable')
    args = parser.parse_args()
    data_dir = args.data_dir

    colmap = args.colmap_exe or find_colmap()
    print(f"Using COLMAP: {colmap}")

    # Feature extraction with shared intrinsics (assume it's the same camera)
    # COLMAP >= 4.0 uses FeatureExtraction prefix, older uses SiftExtraction
    # Try FeatureExtraction first, fallback to SiftExtraction for older versions
    subprocess.run([colmap, 'feature_extractor',
        '--image_path', os.path.join(data_dir, 'images'),
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
        '--FeatureExtraction.use_gpu', '0'], check=True)

    # Feature matching
    subprocess.run([colmap, 'exhaustive_matcher',
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--FeatureMatching.use_gpu', '0'], check=True)

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    subprocess.run([colmap, 'mapper',
        '--image_path', os.path.join(data_dir, 'images'),
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--output_path', os.path.join(data_dir, 'sparse')], check=True)

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    subprocess.run([colmap, 'model_converter',
        '--input_path', os.path.join(data_dir, 'sparse', '0'),
        '--output_path', os.path.join(data_dir, 'sparse', '0_text'),
        '--output_type', 'TXT'], check=True)

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    