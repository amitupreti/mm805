### Instructions to download video

1. Install youtube-dl https://github.com/ytdl-org/youtube-dl

2. Download the youtube video with youtube-dl
    ```shell
    youtube-dl -f 137 "https://www.youtube.com/watch?v=n4-68wdzdAI"
    ```
3. Crop video with FFMEG(if needed)
   ```shell
   ffmpeg -i "Video #1 Calgary to Banff (Trans-Canada Highway)-n4-68wdzdAI.mp4" -ss 00:00:01 -t 00:00:08 -async 1 output.mp4
   ```
4. Generate Optical flow
   ```shell
   python frames2denseflow.py
   ```
5. Generate Mask
   ```shell
   python generate_mask.py
   ```

6. Generate Dataset
   ```shell
   python generate_label.py
   ```
7. For the image, use optical_flow directory
8. Rename the images and seperate into test_train_val
   Edit the prepare_for_training.py(make sure the paths to images are correct, and set train, test, validation size)
   ```shell
   python prepare_for_training.py
   ```
   ```shell