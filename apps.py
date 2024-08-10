import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.video_text import convert_video_to_audio
from core.whisper_text import audio_to_text
from core.openAI_chat import chat_


file_path = 'cropped'
a = os.path.join(os.getcwd(),'audio-files') + f'/{file_path}.wav'
print(os.path.exists(a))
question = "Give a short summary on this video"
print('Started audio extraction............................')
status ,  path_ = convert_video_to_audio(file_path)
print(status,path_,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
if status :
    print('Completed audio extraction............................')
    subtitle = audio_to_text(file_path)
    if subtitle:
        
        chat_(file_path,question)
        
    
    

    
