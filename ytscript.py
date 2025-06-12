import subprocess
import yt_dlp
import speech_recognition as sr
from transformers import pipeline

# Download audio using yt-dlp
def download_audio(url, mp3_path="audio.mp3"):
    print("[1/4] Downloading audio...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': mp3_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("✅ audio.mp3 saved")

# Convert to WAV using ffmpeg
def convert_to_wav(mp3_path="audio.mp3.mp3", wav_path="audio.wav"):
    print("[2/4] Converting to WAV...")
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path], check=True)
    print("✅ audio.wav saved")

# Transcribe with Google Speech Recognition
def transcribe_google(wav_path="audio.wav"):
    print("[3/4] Transcribing via Google STT...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("✅ Transcription complete")
        return text
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""

# Summarize using transformers
def summarize_text(text):
    print("[4/4] Summarizing transcript...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        out = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        summary += out + "\n"
    print("✅ Summary complete")
    return summary

# Main pipeline
if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    download_audio(url)
    convert_to_wav()
    transcript = transcribe_google()
    if not transcript:
        print("⚠️ No transcript available")
    else:
        summary = summarize_text(transcript)
        print("\n=== SUMMARY ===\n", summary)
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print("📝 Summary saved to summary.txt")
