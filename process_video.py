from moviepy.editor import VideoFileClip
from process_image import process_image





output_file = 'output_images/output.mp4'
clip1 = VideoFileClip("input_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(output_file, audio=False)