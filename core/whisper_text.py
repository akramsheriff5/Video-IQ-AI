import whisper ,os


def audio_to_text(path_):
    print("started audio to text extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # Load the Whisper model (you can choose different models like 'base', 'small', 'medium', 'large')
    try:
        model = whisper.load_model("base")

        # Load the audio file and transcribe
        result = model.transcribe(f"./audio-files/{path_}.wav")

        # Save the transcription to a text file
        with open(os.path.join(os.getcwd(),'audio-text') +f"/transcription_{path_}.txt", "w") as f:
            f.write(result["text"])

        # Save as subtitle file (SRT format)
        with open(f"./audio-text/subtitles_{path_}.srt", "w") as srt_file:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Format time in SRT format (hours:minutes:seconds,milliseconds)
                start_time_str = "{:02}:{:02}:{:02},{:03}".format(
                    int(start_time // 3600),
                    int((start_time % 3600) // 60),
                    int(start_time % 60),
                    int((start_time * 1000) % 1000),
                )
                end_time_str = "{:02}:{:02}:{:02},{:03}".format(
                    int(end_time // 3600),
                    int((end_time % 3600) // 60),
                    int(end_time % 60),
                    int((end_time * 1000) % 1000),
                )
                
                # Write the segment to the SRT file
                srt_file.write(f"{i}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n")
        status = True
        print("completed audio to text extraction >>>>>>>>>>>>>>>>>>>>>>>")
    except Exception as e:
        print("error on subtitle extraction ",e)
        status =False
    return status