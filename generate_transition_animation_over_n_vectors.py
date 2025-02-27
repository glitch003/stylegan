# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import sys
import os
import pickle
import numpy as np
import PIL.Image
#import Image
import dnnlib
import dnnlib.tflib as tflib
import config


###################### HOWTO generate video from the generated frames:
###################### ffmpeg -framerate 30 -i animation_%d.png out.mp4
######################

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate.py <path_to_results_dir>")
        return

    model_path = sys.argv[1]

    # Initialize TensorFlow.
    tflib.init_tf()

    seed = 1
    total_runs = 4
    number_of_frames = 240
    frame_step = 1.0/number_of_frames

    # Load pre-trained network.
    #model_path = "./models/karras2019stylegan-celebahq-1024x1024.pkl"
    # model_path = "./results/00005-sgan-painting-1gpu/network-snapshot-008040.pkl"
    #model_path = "./results/00004-sgan-bridge-1gpu/network-snapshot-004705.pkl"
    with open(model_path,"rb") as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    rnd = np.random.RandomState(seed)

    latent_vector1 = rnd.randn(1, Gs.input_shape[1])
    first_vector_ever = latent_vector1.copy()

    total_frames = 1

    for i in range(total_runs):

        if i == total_runs - 1:
            # last run
            latent_vector2 = first_vector_ever
        else:
            latent_vector2 = rnd.randn(1, Gs.input_shape[1])

        x = 0
        for frame_count in range(1,number_of_frames):
            print("generating frame {}".format(total_frames))
            x = x + frame_step
            latent_input = latent_vector1.copy()
            for i in range(512):
                f1 = latent_vector1[0][i]
                f2 = latent_vector2[0][i]
                fnew = (1.0 - x) * f1 + x * f2
                latent_input[0][i] = fnew
            images = Gs.run(latent_input, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)

            # Save image.
            os.makedirs('results/generated', exist_ok=True)
            png_filename = os.path.join('results/generated', 'animation_'+str(total_frames)+'.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

            total_frames += 1

        latent_vector1 = latent_vector2.copy()


if __name__ == "__main__":
    main()
