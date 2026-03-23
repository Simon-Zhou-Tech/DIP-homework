import cv2
import numpy as np
import gradio as gr

# Global variables
points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()

    if img is None:
        image = None
        return None, None, None

    image = img.copy()
    return image, image, image


def draw_points(img, points_src, points_dst):
    if img is None:
        return None

    vis = img.copy()

    # source points: blue
    for pt in points_src:
        cv2.circle(vis, tuple(map(int, pt)), 4, (255, 0, 0), -1)

    # target points: red
    for pt in points_dst:
        cv2.circle(vis, tuple(map(int, pt)), 4, (0, 0, 255), -1)

    # arrow: source -> target
    for i in range(min(len(points_src), len(points_dst))):
        p1 = tuple(map(int, points_src[i]))
        p2 = tuple(map(int, points_dst[i]))
        cv2.arrowedLine(vis, p1, p2, (0, 255, 0), 2, tipLength=0.2)

    return vis


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image

    if image is None:
        return None

    x, y = evt.index

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    return draw_points(image, points_src, points_dst)


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


def run_warping(radius, strength):
    global points_src, points_dst, image

    if image is None:
        return None

    return point_guided_deformation(
        image,
        np.array(points_src, dtype=np.float32),
        np.array(points_dst, dtype=np.float32),
        radius=radius,
        strength=strength
    )


def clear_points():
    global points_src, points_dst, image
    points_src.clear()
    points_dst.clear()

    if image is None:
        return None, None

    return image, image


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

    with gr.Row():
        run_button = gr.Button("Run Warping", variant="primary")
        clear_button = gr.Button("Clear Points")

    input_image.upload(
        fn=upload_image,
        inputs=input_image,
        outputs=[input_image, point_selector, output_image]
    )

    point_selector.select(
        fn=record_points,
        outputs=point_selector
    )

    run_button.click(
        fn=run_warping,
        inputs=[radius_slider, strength_slider],
        outputs=output_image
    )

    clear_button.click(
        fn=clear_points,
        outputs=[point_selector, output_image]
    )

demo.launch()