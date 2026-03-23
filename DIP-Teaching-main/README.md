# Assignment 1 - Image Warping

### 1. Basic Image Geometric Transformation (Scale/Rotation/Translation).
Fill the [Missing Part](run_global_transform.py#L21) of 'run_global_transform.py'.


### 2. Point Based Image Deformation.

Implement MLS or RBF based image deformation in the [Missing Part](run_point_transform.py#L52) of 'run_point_transform.py'.

---

## Implementation of Image Geometric Transformation

First test picture: 

<img src="test1.jpg" alt="alt text" width="800">

Second test picture:

<img src="test2.jpg" alt="alt text" width="800">


## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Fill Part

in basic transformation part:
```point
# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
        ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1) 平移到图像中心
    T_to_center = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 2) 缩放
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 3) 旋转
    theta = np.deg2rad(rotation)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=np.float32)

    # 4) 水平翻转（围绕图像中心）
    if flip_horizontal:
        F = np.array([
            [-1, 0, 0],
            [0,  1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
    else:
        F = np.eye(3, dtype=np.float32)

    # 5) 从中心移回原位置
    T_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 6) 最终平移
    T_translate = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 组合变换：
    # 先移到中心 -> 缩放 -> 旋转 -> 翻转 -> 移回去 -> 平移
    M = T_translate @ T_back @ F @ R @ S @ T_to_center

    # cv2.warpAffine 需要 2x3 矩阵
    transformed_image = cv2.warpAffine(
        image,
        M[:2, :],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return transformed_image
```
in point guided transformation part:

part1:
```python

def local_translation_warp(img, src_pt, dst_pt, radius=40, strength=1.0):
    """
    单对控制点的局部平移形变（backward warping）
    src_pt: 原位置
    dst_pt: 目标位置
    radius: 影响半径
    strength: 强度系数
    """
    h, w = img.shape[:2]

    src_x, src_y = float(src_pt[0]), float(src_pt[1])
    dst_x, dst_y = float(dst_pt[0]), float(dst_pt[1])

    # backward warping: 输出图 dst 附近回采样到 src 附近
    move_x = (src_x - dst_x) * strength
    move_y = (src_y - dst_y) * strength

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    dx = xx - dst_x
    dy = yy - dst_y
    dist = np.sqrt(dx * dx + dy * dy)

    mask = dist < radius

    # 更柔和一点的衰减函数
    weight = np.zeros_like(xx, dtype=np.float32)
    t = 1.0 - dist[mask] / radius
    weight[mask] = t * t * (3.0 - 2.0 * t)   # smoothstep 风格

    map_x = xx.copy()
    map_y = yy.copy()

    map_x[mask] = xx[mask] + move_x * weight[mask]
    map_y[mask] = yy[mask] + move_y * weight[mask]

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped


def point_guided_deformation(image, source_pts, target_pts, radius=40, strength=1.0):
    """
    对多对控制点依次执行局部 warp
    """
    if image is None:
        return None

    warped = image.copy()

    if len(source_pts) == 0 or len(source_pts) != len(target_pts):
        return warped

    for src_pt, dst_pt in zip(source_pts, target_pts):
        warped = local_translation_warp(
            warped,
            src_pt,
            dst_pt,
            radius=radius,
            strength=strength
        )

    return warped

```
part2:
```python

with gr.Blocks() as demo:
    gr.Markdown("## Point-Guided Face Warp Demo")

    with gr.Row():
        with gr.Column(scale=3):
            input_image = gr.Image(
                label="上传图片",
                type="numpy",
                interactive=True
            )

            point_selector = gr.Image(
                label="点击选择控制点和目标点（source / target 交替点击）",
                type="numpy",
                interactive=True
            )

        with gr.Column(scale=2):
            output_image = gr.Image(
                label="变换结果",
                type="numpy"
            )

            radius_slider = gr.Slider(
                minimum=10,
                maximum=200,
                value=70,
                step=1,
                label="Radius（影响范围）"
            )

            strength_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Strength（形变强度）"
            )

```


## Running

To run basic transformation, run:

```python
python run_global_transform.py
```

To run point guided transformation, run:

```python
python run_point_transform.py
```

## Results (need add more result images)
### Basic Transformation
<img src="run_global_transform-checkpoint_result.gif" alt="alt text" width="800">

<img src="run_global_transform-checkpoint_result2.gif" alt="alt text" width="800">

### Point Guided Deformation:
<img src="run_point_transform-checkpoint_result.gif" alt="alt text" width="800">

<img src="run_point_transform-checkpoint_result2.gif" alt="alt text" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
