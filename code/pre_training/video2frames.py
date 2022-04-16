import cv2
import os

# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames and github copilot
def video_to_frames(video_path, output_path):
    """
    This function reads a video and then saves each frame of video into output_path
    :param video_path: path of video
    :param output_path: path to save frames
    :return: None
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    while success:
        if count > 20:
            # skip first 20 frames as they are normally black screen or intro
            cv2.imwrite(f'{output_path}/frame{count}.jpg', image)  # save frame as JPEG file
        success, image = vidcap.read()
        print(f'Read a new frame: {success} Count: {count}')
        count += 1


if __name__ == '__main__':
    video_to_frames('video.mp4', './frames')
