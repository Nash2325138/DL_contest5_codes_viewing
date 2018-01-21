import numpy as np

def get_nice_session(fraction=0.25):
    import tensorflow as tf
    
    config = tf.ConfigProto()
    if fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    else:
        config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def make_anim(images, fps=60, true_image=False):
    import moviepy.editor as mpy

    duration = len(images) / fps
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip
