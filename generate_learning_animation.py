# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys
import glob

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate.py <path_to_results_dir>")
        return

    # Initialize TensorFlow.
    tflib.init_tf()

    result_dir = sys.argv[1]

    pkls = glob.glob(os.path.join(result_dir,"network-snapshot-*.pkl"))
    pkls.sort()

    seed = 1
    i = 1
    for file in pkls:
        print("Loading {} of {} with name {}".format(i, len(pkls), file))

        # Load pre-trained network.
        # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(open(file, 'rb'), encoding='latin1')
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        Gs.print_layers()

        # Pick latent vector.
        rnd = np.random.RandomState(seed)
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

        # Save image.
        output_dir = os.path,join(result_dir, 'generated_learning')
        os.makedirs(output_dir, exist_ok=True)
        png_filename = os.path.join(output_dir, 'frame_{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

        # prevent memory leak
        del _G
        del _D
        del Gs

        i += 1

if __name__ == "__main__":
    main()
