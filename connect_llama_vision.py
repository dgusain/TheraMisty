import os
import re
import time
import json
import gc
import logging
from typing import List, Optional, Dict

import requests
import torch
import wave
import vosk
import speech_recognition as sr
from transformers import MllamaForConditionalGeneration, AutoProcessor
from mistyPy.Robot import Robot
from mistyPy.Events import Events

# Configuration Constants
class Config:
    MAIN_CACHE_DIR: str = "/home/dgusain/misty/Huggingface/"  # Main cache directory
    
    # Updated model name to Llama-3.2-Vision-11B for text generation
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    LOCAL_MODEL_DIR: str = os.path.join(MAIN_CACHE_DIR, MODEL_NAME)  # Derived local model directory
    AUDIO_SAVE_PATH: str = "/home/dgusain/misty/Python-SDK/mistyPy/misty_user_recording.wav"
    VOSK_MODEL_PATH: str = "model"
    MISTY_IP: str = "67.20.193.16"
    AUDIO_URL: str = f"http://{MISTY_IP}/api/audio?fileName=capture_Dialogue.wav&base64=false"
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    MAX_INPUT_LENGTH: int = 4096  # Increased for larger context
    MAX_NEW_TOKENS: int = 100  # Adjusted for more extensive responses
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50
    SLEEP_DURATION: float = 5.50  # seconds
    GPU_ID: Optional[int] = int(os.getenv("GPU_ID", 0))  # Default to GPU 0. Adjust as needed.
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")  # Optional: For private models

# Initialize Logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class MistyAssistant:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.recognizer = sr.Recognizer()

        # Set device based on GPU_ID
        if torch.cuda.is_available():
            if Config.GPU_ID is not None:
                try:
                    self.device = torch.device(f"cuda:{Config.GPU_ID}")
                    logger.info(f"Using GPU {Config.GPU_ID}")
                except AssertionError as e:
                    logger.error(f"GPU ID {Config.GPU_ID} is not available. Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
                logger.info("Using default CUDA device")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")

        self.model = None
        self.processor = None
        self.misty = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are Misty, a social robot designed to assist users."}
        ]

    def load_model(self):
        try:
            logger.info(f"Loading Hugging Face model: {Config.MODEL_NAME}...")
            start_time = time.time()

            # Ensure the local cache directory exists
            os.makedirs(Config.MAIN_CACHE_DIR, exist_ok=True)
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=Config.MAIN_CACHE_DIR,
                use_auth_token=Config.HUGGINGFACE_TOKEN,  # Optional: For private models
                torch_dtype=torch.float16,  # Use float16 for efficiency with large models
            ).half().to(self.device)

            # Load the processor with caching
            self.processor = AutoProcessor.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=Config.MAIN_CACHE_DIR,
                use_auth_token=Config.HUGGINGFACE_TOKEN  # Optional: For private models
            )

            # Load the model with caching

            logger.info("Model and processor loaded successfully.")
            self.timings['Model loading'] = time.time() - start_time

        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

    @staticmethod
    def extract_misty_response(response: str) -> str:
        """
        Extracts the first line that starts with 'Misty:' from the response.
        """
        pattern = r'^(?:System|Misty):\s*(.*)'
        matches = re.findall(pattern, response, re.MULTILINE)
        if len(matches) >= 1:
            return matches[0].strip()
        else:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

    @staticmethod
    def apply_chat_template(messages: List[Dict[str, str]]) -> str:
        """
        Formats the messages list into a single string suitable for model input.
        """
        role_map = {
            "system": "System",
            "user": "User",
            "assistant": "Misty"
        }
        formatted_messages = ""
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            prefix = role_map.get(role, "User")
            formatted_messages += f"{prefix}: {content}\n"
        return formatted_messages

    def generate_response_transformers(self) -> str:
        """
        Generates a response from the model based on the conversation history.
        Extracts only the first response from Misty.
        """
        try:
            formatted_prompt = self.apply_chat_template(self.messages)
            logger.debug("Formatted prompt for model:\n%s", formatted_prompt)
            logger.info("Generating response...")

            # Prepare inputs using AutoProcessor without image
            inputs = self.processor(
                text=formatted_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

            response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug("Raw Generated response: %s", response)

            # Extract only Misty's first response line
            misty_response = self.extract_misty_response(response)
            logger.info("Extracted Misty's response: %s", misty_response)

            return misty_response
        except Exception as e:
            logger.exception(f"Failed to generate response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def download_audio_file(self, audio_url: str) -> bool:
        """
        Downloads the audio file from the specified URL.
        """
        logger.info("Sending request to download audio from %s", audio_url)
        try:
            response = requests.get(audio_url, timeout=10)
            response.raise_for_status()
            with open(Config.AUDIO_SAVE_PATH, 'wb') as audio_file:
                audio_file.write(response.content)
            logger.info("Audio file saved as: %s", Config.AUDIO_SAVE_PATH)
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to download audio file: {e}")
            return False
        except IOError as e:
            logger.error(f"IO error when saving audio file: {e}")
            return False

    def transcribe_audio(self) -> Optional[str]:
        """
        Transcribes the downloaded audio file using Google Speech Recognition.
        """
        logger.info("Starting voice transcription using Google Speech Recognition.")
        try:
            with sr.AudioFile(Config.AUDIO_SAVE_PATH) as source:
                audio_data = self.recognizer.record(source)
            transcript = self.recognizer.recognize_google(audio_data)
            logger.info("Transcribed text: %s", transcript)
            return transcript
        except sr.RequestError as e:
            logger.error(f"API request error during transcription: {e}")
        except sr.UnknownValueError:
            logger.error("Google Speech Recognition could not understand audio.")
        except Exception as e:
            logger.exception(f"Unexpected error during transcription: {e}")
        return None

    def capture_speech_callback(self, data):
        """
        Callback function for voice recording event.
        Currently a placeholder for future implementation.
        """
        logger.debug("Voice capture callback triggered.")

    def start_voice_capture(self):
        """
        Registers the voice capture event and initiates speech capture.
        """
        try:
            logger.info("Registering voice capture event with Misty.")
            self.misty.register_event(
                event_type=Events.VoiceRecord,
                event_name="AudioCallbackEvent",
                callback_function=self.capture_speech_callback
            )
            capture_start = time.time()
            self.misty.capture_speech()
            capture_duration = time.time() - capture_start
            Config.SLEEP_DURATION = max(capture_duration, Config.SLEEP_DURATION)
            logger.info("Initiated speech capture. Waiting for %s seconds.", Config.SLEEP_DURATION)
            time.sleep(Config.SLEEP_DURATION)
        except Exception as e:
            logger.exception(f"Failed to capture voice: {e}")
            raise

    def transcribe_with_vosk(self) -> Optional[str]:
        """
        Transcribes the audio file using Vosk speech recognition.
        """
        logger.info("Starting voice transcription using Vosk.")
        if not os.path.exists(Config.AUDIO_SAVE_PATH):
            logger.error("Audio file does not exist at %s", Config.AUDIO_SAVE_PATH)
            return None

        try:
            model = vosk.Model(Config.VOSK_MODEL_PATH)
            with wave.open(Config.AUDIO_SAVE_PATH, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100]:
                    logger.error("Audio file must be WAV format mono PCM.")
                    return None

                rec = vosk.KaldiRecognizer(model, wf.getframerate())
                transcript = ""
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        transcript += result.get('text', '') + " "

                result = json.loads(rec.FinalResult())
                transcript += result.get('text', '')
            
            transcript = transcript.strip()
            logger.info("Vosk Transcribed text: %s", transcript)
            return transcript
        except Exception as e:
            logger.exception(f"Failed to transcribe audio with Vosk: {e}")
            return None

    def run(self):
        """
        Main execution method for the Misty Assistant.
        """
        try:
            overall_start_time = time.time()

            # Load model
            self.load_model()

            # Initialize Misty robot
            self.misty = Robot(Config.MISTY_IP)
            logger.info("Connected to Misty at %s", Config.MISTY_IP)

            # Start voice capture
            start_time = time.time()
            self.start_voice_capture()
            self.timings['Misty speech capture'] = time.time() - start_time

            # Download the audio file
            start_time = time.time()
            if self.download_audio_file(Config.AUDIO_URL):
                self.timings['Audio file download'] = time.time() - start_time
            else:
                logger.error("Audio file download failed. Exiting.")
                return

            # Voice transcription using Google
            start_time = time.time()
            user_text = self.transcribe_audio()
            if "image" in user_text:
                
            if user_text:
                self.timings['Voice transcription'] = time.time() - start_time
                self.messages.append({"role": "user", "content": user_text})
                logger.info(f"User said: {user_text}")
            else:
                logger.error("Voice transcription failed. Exiting.")
                return

            # Alternatively, you can use Vosk for transcription
            # Uncomment the lines below to use Vosk instead of Google
            # start_time = time.time()
            # user_text = self.transcribe_with_vosk()
            # if user_text:
            #     self.timings['Voice transcription (Vosk)'] = time.time() - start_time
            #     self.messages.append({"role": "user", "content": user_text})
            #     logger.info(f"User said: {user_text}")
            # else:
            #     logger.error("Vosk transcription failed. Exiting.")
            #     return

            # Generate response using the model
            start_time = time.time()
            response = self.generate_response_transformers()
            self.timings['LLM execution'] = time.time() - start_time

            # Append assistant's response to messages
            self.messages.append({"role": "assistant", "content": response})

            # Misty speaks the response
            start_time = time.time()
            logger.info("Misty's response: %s", response)
            self.misty.speak(response, None, None, None, True, "tts-content")
            self.timings['Misty Response processing'] = time.time() - start_time

            overall_end_time = time.time()
            self.timings['Total time taken'] = overall_end_time - overall_start_time
            self.timings['Latency'] = (
                self.timings['Total time taken'] -
                self.timings.get('Misty Response processing', 0) -
                self.timings.get('Model loading', 0)
            )
            self.timings['Latency excluding speech capture'] = self.timings['Latency'] - self.timings['Misty speech capture']

            # Log timings
            for process, duration in self.timings.items():
                logger.info(f"{process}: {duration:.2f} seconds")

        except Exception as ex:
            logger.exception(f"An error occurred during execution: {ex}")
        finally:
            # Cleanup resources
            if self.misty:
                self.misty.unregister_all_events()
                logger.info("Unregistered all events from Misty.")

            if self.device.type == "cuda" and self.model:
                torch.cuda.empty_cache()
                del self.model
                gc.collect()
                logger.info("Cleaned up CUDA resources.")

            logger.info("Misty Assistant has terminated.")

if __name__ == "__main__":
    assistant = MistyAssistant()
    assistant.run()
