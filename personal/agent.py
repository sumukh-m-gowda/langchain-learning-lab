"""
Voice-controlled laptop agent (Windows)
Mic -> Speech-to-Text -> LangGraph agent (Gemini) -> tools -> spoken reply
"""

import os
import subprocess

import pyttsx3
import screen_brightness_control as sbc
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()  # loads GOOGLE_API_KEY from .env

# ---------------------------------------------------------------
# 1. TOOLS  (the real actions the agent can perform)
# ---------------------------------------------------------------

@tool
def set_brightness(level: int) -> str:
    """Set the screen brightness. level must be a number from 0 to 100."""
    level = max(0, min(100, level))
    sbc.set_brightness(level)
    return f"Brightness set to {level}%"


@tool
def get_brightness() -> str:
    """Get the current screen brightness percentage."""
    current = sbc.get_brightness()
    value = current[0] if isinstance(current, list) else current
    return f"Current brightness is {value}%"


@tool
def open_settings() -> str:
    """Open the Windows Settings app."""
    os.startfile("ms-settings:")
    return "Settings opened"


# Only apps in this whitelist can be opened (safer than running any command)
ALLOWED_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "chrome": "chrome.exe",
}


@tool
def open_app(name: str) -> str:
    """Open an app by name. Allowed: notepad, calculator, paint, file explorer, chrome."""
    exe = ALLOWED_APPS.get(name.lower().strip())
    if not exe:
        return f"I can't open '{name}'. Allowed apps: {', '.join(ALLOWED_APPS)}"
    subprocess.Popen(exe, shell=True)
    return f"Opened {name}"


TOOLS = [set_brightness, get_brightness, open_settings, open_app]

# ---------------------------------------------------------------
# 2. LLM + AGENT
# ---------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # check AI Studio if name changes
agent = create_react_agent(
    llm,
    TOOLS,
    prompt="You are a helpful laptop assistant. Use the tools to control the laptop. "
           "Keep replies very short, since they are spoken aloud.",
)

# ---------------------------------------------------------------
# 3. VOICE: speech-to-text and text-to-speech
# ---------------------------------------------------------------

recognizer = sr.Recognizer()
engine = pyttsx3.init()


def listen() -> str | None:
    """Record from the mic and return the recognized text (or None)."""
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Couldn't understand, try again.")
    except sr.RequestError:
        print("Speech service unavailable (check internet).")
    return None


def speak(text: str) -> None:
    engine.say(text)
    engine.runAndWait()


def extract_text(content) -> str:
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return " ".join(parts)


# ---------------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------------

def main():
    print("Laptop agent ready. Say 'exit' or 'stop' to quit.")
    print("Tip: press Enter to type instead of speaking.\n")
    while True:
        mode = input("Press Enter to speak (or type a command): ").strip()
        text = mode if mode else listen()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "stop"):
            speak("Goodbye")
            break

        try:
            result = agent.invoke({"messages": [("user", text)]})
            reply = extract_text(result["messages"][-1].content)
        except Exception as e:
            reply = f"Something went wrong: {e}"

        print("Agent:", reply)
        speak(reply)


if __name__ == "__main__":
    main()