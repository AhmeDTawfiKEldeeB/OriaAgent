import argparse
import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.core.exceptions import TextToSpeechError
from src.modules.speech.text_to_speech import TextToSpeech


async def test_real_tts(
    text: str = "Hello Ahmed, i am oria what about you today good or not.",
    custom_voice_id: str | None = None,
    output_filename: str = "test_oria.mp3",
) -> bool:
    """Run real text-to-speech synthesis using configured ElevenLabs credentials."""
    print("=" * 65)
    print(" ElevenLabs Text-To-Speech (TTS) Real Test")
    print("=" * 65)

    settings = get_settings()

    # Check credentials
    api_key = settings.ELEVENLABS_API_KEY or os.getenv("ELEVENLABS_API_KEY")
    voice_id =settings.ELEVENLABS_VOICE_ID or os.getenv("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        print("\n[!] ERROR: Missing ElevenLabs credentials.")
        print("    Please ensure ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are set in your .env file.")
        print(f"    ELEVENLABS_API_KEY : {'[SET]' if api_key else '[MISSING]'}")
        print(f"    ELEVENLABS_VOICE_ID: {'[SET]' if voice_id else '[MISSING]'}")
        return False

    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"\n[1/3] Initializing TTS client...")
    print(f"      API Key : {masked_key}")
    print(f"      Voice ID: {voice_id}")

    try:
        tts = TextToSpeech(api_key=api_key, voice_id=voice_id)
        print("      Client successfully initialized.")
    except Exception as e:
        print(f"\n[!] Failed to initialize TextToSpeech: {e}")
        return False

    print(f"\n[2/3] Synthesizing speech...")
    print(f"      Text: \"{text}\"")

    audio_bytes = None
    try:
        audio_bytes = await tts.synthesize(text)
        print(f"      Generated audio size: {len(audio_bytes):,} bytes")
    except TextToSpeechError as e:
        err_msg = str(e)
        if "paid_plan_required" in err_msg or "402" in err_msg:
            print("\n[!] ElevenLabs Plan Restriction:")
            print(f"    Voice ID '{voice_id}' is a community library voice.")
            print("    ElevenLabs free accounts cannot use library voices via API (HTTP 402 Paid Plan Required).")
            print("\n[i] Recommended Free Voices for Oria:")
            print("    Female Voices:")
            print("      - Jessica (Playful, Bright, Warm) : cgSgspJ2msm6clMCkdW9")
            print("      - Sarah   (Mature, Reassuring)   : EXAVITQu4vr4xnSDxMaL")
            print("      - Lily    (Velvety, Warm)        : pFZP5JQG7iQjIQuC4Bku")
            print("      - Alice   (Clear, Engaging)      : Xb7hH8MSUJpSbSDYk0k2")
            print("      - Bella   (Professional, Warm)   : hpp4J3VqNfWAUOO0d1Us")
            print("    Male Voices:")
            print("      - George  (Warm Storyteller)     : JBFqnCBsd6RMkjVDRZzb")
            print("      - Roger   (Laid-back, Resonant)  : CwhRBWXzGAHq8TQ4Fs17")
            print("      - Charlie (Deep, Confident)      : IKne3meq5aSn9XLyUdCD")
            print("\n[TIP] Set one of the above IDs in your .env file, for example:")
            print("      ELEVENLABS_VOICE_ID=cgSgspJ2msm6clMCkdW9")
            return False
        else:
            print(f"\n[!] Synthesis failed: {e}")
            return False
    except Exception as e:
        print(f"\n[!] Unexpected error during synthesis: {e}")
        return False

    if not audio_bytes:
        print("\n[!] No audio data was generated.")
        return False

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_filename))
    print(f"\n[3/3] Saving audio to disk...")
    try:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        size_kb = len(audio_bytes) / 1024
        print(f"      Audio saved successfully to: {output_path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"\n[!] Failed to save audio file: {e}")
        return False

    print("\n" + "=" * 65)
    print(" SUCCESS: Text-to-speech synthesis completed successfully!")
    print(f" File: {output_path}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ElevenLabs Text-to-Speech synthesis.")
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to synthesize",
    )
    parser.add_argument("--voice-id", "-v", default=None, help="Custom ElevenLabs Voice ID to use")
    parser.add_argument("--output", "-o", default=None, help="Output audio filename")
    args = parser.parse_args()

    call_kwargs = {}
    if args.text is not None:
        call_kwargs["text"] = args.text
    if args.voice_id is not None:
        call_kwargs["custom_voice_id"] = args.voice_id
    if args.output is not None:
        call_kwargs["output_filename"] = args.output

    asyncio.run(test_real_tts(**call_kwargs))