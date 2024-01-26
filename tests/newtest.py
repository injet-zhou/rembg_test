from rembg import remove
from rembg.session_factory import new_session
from PIL import Image
from multiprocessing import Pool
from time import perf_counter
import cProfile
import os
from datetime import datetime

session = new_session('isnet-general-use')
img = Image.open('./pic.png')

def split_image(original_image):

    # 获取图片尺寸
    width, height = original_image.size

    # 切割成四等分
    top_left = original_image.crop((0, 0, width // 2, height // 2))
    top_right = original_image.crop((width // 2, 0, width, height // 2))
    bottom_left = original_image.crop((0, height // 2, width // 2, height))
    bottom_right = original_image.crop((width // 2, height // 2, width, height))

    return top_left, top_right, bottom_left, bottom_right

def merge_images(images):
    # 获取每个部分的尺寸
    width = max(img.width for img in images)
    height = max(img.height for img in images)

    # 创建一个新的图片，将切割后的四个部分拼接回去
    new_image = Image.new("RGB", (width * 2, height * 2))
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (width, 0))
    new_image.paste(images[2], (0, height))
    new_image.paste(images[3], (width, height))
    new_image.save("new.png")

def myremove(preconditioner: str, image: Image.Image):

    result = remove(
        image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=15,
        session=session,
        preconditioner=preconditioner,
    )
    return result


def remove_async():
    start = perf_counter()
    preconditioner = 'ichol'
    images = split_image(img)
    with Pool(4) as p:
        results = p.starmap(myremove, [(preconditioner, image) for image in images])
    merge_images(results)
    end = perf_counter()
    print(f"Time: {end - start}s")

def just_remove():
    start = perf_counter()
    preconditioner = 'ichol'
    result = myremove(preconditioner, img)
    end = perf_counter()
    result.save("new2.png")
    print(f"Time: {end - start}s")


def with_profile(fn):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort='cumtime')
        profiler.dump_stats('newtest.prof')
    return wrapper

def get_images_from_dir(dir:str) -> list:
    return [os.path.join(dir, f) for f in os.listdir(dir)]

def get_filename_without_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def batch_test():
    src_dir = "./testsrc"
    dst_base_dir = "./results"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dst_dir = os.path.join(dst_base_dir, timestamp)
    os.makedirs(dst_dir)
    images = get_images_from_dir(src_dir)
    for image in images:
        start = perf_counter()
        preconditioner = 'ichol'
        img = Image.open(image)
        result = myremove(preconditioner, img)
        new_file_name = get_filename_without_ext(image) + ".png"
        result.save(os.path.join(dst_dir, new_file_name))
        end = perf_counter()
        print(f"========= processing {image} cost: {end - start}s")


if __name__ == '__main__':
    batch_test()
