import argparse
from pytube import YouTube
import os 
#execute with conda base 
#python yt_vid_audio_download.py -v https://www.youtube.com/watch\?v\=_SloSMr-gFI
VIDEO_SAVE_DIRECTORY = "data/yt_videos"
AUDIO_SAVE_DIRECTORY = "data/yt_audio"

if not os.path.exists(VIDEO_SAVE_DIRECTORY):
        os.makedirs(VIDEO_SAVE_DIRECTORY)
        print(f"Directory '{VIDEO_SAVE_DIRECTORY}' created.")
else:
        print(f"Directory '{VIDEO_SAVE_DIRECTORY}' already exists.")
if not os.path.exists(AUDIO_SAVE_DIRECTORY):
        os.makedirs(AUDIO_SAVE_DIRECTORY)
        print(f"Directory '{AUDIO_SAVE_DIRECTORY}' created.")
else:
        print(f"Directory '{VIDEO_SAVE_DIRECTORY}' already exists.")

def download(video_url):
    video = YouTube(video_url)
    video = video.streams.get_highest_resolution()

    try:
        video.download(VIDEO_SAVE_DIRECTORY)
    except:
        print("Failed to download video")

    print("video was downloaded successfully")

def download_audio(video_url):
    video = YouTube(video_url)
    audio = video.streams.filter(only_audio = True).first()

    try:
        audio.download(AUDIO_SAVE_DIRECTORY)
    except:
        print("Failed to download audio")

    print("audio was downloaded successfully")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required = True, help = "URL to youtube video")
    ap.add_argument("-a", "--audio", required = False, help = "audio only", action = argparse.BooleanOptionalAction)
    args = vars(ap.parse_args())

    if args["audio"]:
        download_audio(args["video"])
    else:
        download(args["video"])

#!ffmpeg -ss 378 -i fed_meeting.mp4 -t 2715 fed_meeting_trimmed.mp4
# We can do some additional processing on the audio file should we choose. 
# I want to ignore any additional sound and speech after Jerome Powell speaks. 
# So we'll use ffmpeg to do this. 
# The command will start the audio file at the 375 second mark where he starts with good afternoon, continue for 2715 seconds, and chop off the rest of the audio. The result will be saved in a new file called fed_meeting_trimmed.mp4.