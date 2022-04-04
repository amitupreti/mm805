

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