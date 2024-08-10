import moviepy.editor as mp 
import speech_recognition as sr 
import os
  
def convert_video_to_audio(path_):
    try:
        # Load the video 
        video = mp.VideoFileClip(os.path.join(os.getcwd(),'videos') + f"/{path_}.mp4") 
        
        # Extract the audio from the video 
        
        audio_file = video.audio 
        audio_file.write_audiofile(os.path.join(os.getcwd(),'audio-files') + f'/{path_}.wav') 
        status = True
    except Exception as e:
        print("Audio extraction error....",e)
        status = False
    
    return status , path_
    
 