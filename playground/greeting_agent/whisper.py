import torch
import whisper
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
import os
import contextlib
import wave
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

def transcribe_with_diarization(audio_path, output_file):
    """
    Transcribes an audio file with speaker diarization.

    Args:
        audio_path (str): Path to the input audio file.
        output_file (str): Path to save the transcription.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    print("Loading diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        pipeline.to(torch.device(device))
    except Exception as e:
        print(f"Error loading pyannote pipeline: {e}")
        print("Please ensure you have a valid Hugging Face token (HF_TOKEN)")
        print("and accepted the model's user agreement on Hugging Face.")
        return

    print(f"Running diarization on {audio_path}...")
    diarization = pipeline(audio_path, num_speakers=2)

    print("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base", device=device)

    print("Loading audio file for transcription...")
    try:
        audio_waveform = whisper.load_audio(audio_path)
        sample_rate = whisper.audio.SAMPLE_RATE # 16000
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    print("Transcribing segments and writing to file...")
    
    all_segments = []
    for segment, track_id, label in diarization.itertracks(yield_label=True):
        all_segments.append({
            'start': segment.start,
            'end': segment.end,
            'label': label
        })
    
    if not all_segments:
        print("No speaker segments found by pyannote.")
        return
        
    all_segments.sort(key=lambda x: x['start'])
    
    merged_segments = []
    current_segment = all_segments[0].copy()

    for next_seg in all_segments[1:]:
        if (next_seg['label'] == current_segment['label'] and 
            next_seg['start'] - current_segment['end'] < 0.1):
            current_segment['end'] = next_seg['end']
        else:
            merged_segments.append(current_segment)
            current_segment = next_seg.copy()
    
    merged_segments.append(current_segment)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Transcription of {os.path.basename(audio_path)}\n\n")
        
        for i, segment in enumerate(merged_segments):
            start_time = segment['start']
            end_time = segment['end']
            label = segment['label']
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            segment_audio = audio_waveform[start_sample:min(end_sample, len(audio_waveform))]

            result = whisper_model.transcribe(segment_audio, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            if text:
                start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
                end_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:06.3f}"
                
                line = f"[{start_str} --> {end_str}] {label}: {text}\n"
                
                print(line, end="")
                f.write(line)

    print(f"\nTranscription complete! Saved to {output_file}")

if __name__ == "__main__":
    INPUT_AUDIO = "/home/dedsec995/Downloads/jeff.wav" 
    OUTPUT_TRANSCRIPT = "/home/dedsec995/out.txt"
    transcribe_with_diarization(INPUT_AUDIO, OUTPUT_TRANSCRIPT)