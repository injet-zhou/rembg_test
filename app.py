import gradio as gr
from pymatting.preconditioner.ichol import ichol
from rembg.session_factory import new_session
from time import perf_counter
from rembg.bg import remove

models = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
]

session_cache = {}

def hook_ichol(
        discard_threshold=1e-4,
    shifts=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5],
):
    def my_ichol(A):
        return ichol(
            A,
            discard_threshold=discard_threshold,
            shifts=shifts,
        )
    return my_ichol

def parse_float(text, title):
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{title} 必须为数字或科学计数法表示的数字")

def parse_shifts(shift_txt):
    try:
        return [float(shift) for shift in shift_txt.split(",")]
    except ValueError:
        raise ValueError("Shifts 必须为逗号分隔的数字或科学计数法表示的数字")

def perf_html(load_cost, remove_cost):
    return f"""
    <h3>Info</h3>
    <p>加载模型耗时: {load_cost:.2f}s</p>
    <p>去除背景耗时: {remove_cost:.2f}s</p>
    """

def remove_bg(
    image,
    alpha_matting,
    alpha_matting_foreground_threshold,
    alpha_matting_background_threshold,
    alpha_matting_erode_size,
    model,
    only_mask,
    discard_threshold,
    shifts,
    epsilon,
):
    print(f"""remove background using following parameters:
    alpha_matting: {alpha_matting}
    alpha_matting_foreground_threshold: {alpha_matting_foreground_threshold}
    alpha_matting_background_threshold: {alpha_matting_background_threshold}
    alpha_matting_erode_size: {alpha_matting_erode_size}
    model: {model}
    only_mask: {only_mask}
    discard_threshold: {discard_threshold}
    shifts: {shifts}
    epsilon: {epsilon}
""")
    if image is None:
        raise ValueError("请上传图片")
    load_model_start = perf_counter()
    if model not in session_cache:
        session_cache[model] = new_session(model)
    session = session_cache[model]
    load_model_end = perf_counter()

    discard_threshold = parse_float(discard_threshold, "Discard threshold") if discard_threshold else 1e-4
    epsilon = parse_float(epsilon, "Epsilon") if epsilon else 1e-7
    shifts = parse_shifts(shifts) if shifts else [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5]

    remove_bg_start = perf_counter()
    removed = remove(
        image,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
        session=session,
        preconditioner=hook_ichol(
            discard_threshold=discard_threshold,
            shifts=shifts,
        ),
        epsilon=epsilon,
    )
    remove_bg_end = perf_counter()
    html = perf_html(load_model_end - load_model_start, remove_bg_end - remove_bg_start)

    return [(removed, "generated")], html


with gr.Blocks() as demo:
    gr.Markdown("## Rembg custom demo")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Image")
            alpha_matting = gr.Checkbox(label="Alpha Matting")
            alpha_matting_foreground_threshold = gr.Slider(
                0, 255, 240, label="Alpha Matting Foreground Threshold"
            )
            alpha_matting_background_threshold = gr.Slider(
                0, 255, 15, label="Alpha Matting Background Threshold"
            )
            alpha_matting_erode_size = gr.Slider(
                0, 255, 15, label="Alpha Matting Erode Size"
            )
            model = gr.Dropdown(models, label="Model", value="isnet-general-use")
            only_mask = gr.Checkbox(label="Only Mask")
            discard_threshold = gr.Textbox(
                label="Discard threshold",
                value="1e-4",
                info="计算Cholesky分解时，丢弃的最小值。默认值为1e-4，越大，shift尝试的值就越多，但是也会增加计算时间（迭代次数）。"
                )
            shifts = gr.Textbox(
                label="Shifts",
                value="0.0,0.0001,0.001,0.01,0.1,0.5,1.0,10.0,100,1000.0,10000.0,100000.0",
                info="用于尝试正则化的值，第一个值越大，收敛可能会更快。\n推荐值 0.0001,0.001,0.01,0.1,0.5,1.0,10.0,100,1000.0,10000.0,100000.0"
            )
            epsilon = gr.Textbox(
                label="Epsilon",
                value="1e-7",
                info="正则化强度，pymatting中默认为1e-7，较大的正则化强度可以提高收敛速度, 但是alpha matte会有一定的损失.\n可尝试1e-5或1e-6获得更快的速度"
                )

            submit = gr.Button("Submit")
        with gr.Column():
            output = gr.Gallery(label="Output", show_label=False, columns=1, preview=True)
            generated_text = gr.HTML("<h3>Info</h3>")

        submit.click(
                fn=remove_bg,
                inputs=[
                    image,
                    alpha_matting,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                    model,
                    only_mask,
                    discard_threshold,
                    shifts,
                    epsilon,
                ],
                outputs=[output,generated_text],
            )

demo.launch()
