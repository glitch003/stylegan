ffmpeg -framerate 30 -i generated/animation_%d.png -c:v libx264 -pix_fmt yuv420p -r 30 out.mp4