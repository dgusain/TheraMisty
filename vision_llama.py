import os
import re
import time
import json
import gc
import logging
from typing import List, Optional, Dict
from PIL import Image
import io
import requests
import torch
import wave
#import vosk
import speech_recognition as sr
from transformers import MllamaForConditionalGeneration, AutoProcessor
from mistyPy.Robot import Robot
from mistyPy.Events import Events

from flash_attn.flash_attn_interface import flash_attn_func
from threading import Timer

# Configuration Constants
class Config:
    MAIN_CACHE_DIR: str = "/home/dgusain/misty/Huggingface/"  # Main cache directory
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Updated model name
    LOCAL_MODEL_DIR: str = os.path.join(MAIN_CACHE_DIR, MODEL_NAME)  # Derived local model directory
    AUDIO_SAVE_PATH: str = "/home/dgusain/misty/Python-SDK/mistyPy/misty_user_recording.wav"
    IMAGE_SAVE_PATH: str = "/home/dgusain/misty/Python-SDK/mistyPy/user_pic.jpg"
    MISTY_IP: str = "67.20.193.16"
    AUDIO_URL: str = f"http://{MISTY_IP}/api/audio?fileName=capture_Dialogue.wav&base64=false"
    IMAGE_URL: str = f"http://{MISTY_IP}/api/cameras/rgb?base64=false&fileName=user_pic&displayOnScreen=false&overwriteExisting=false"
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    MAX_INPUT_LENGTH: int = 2048
    MAX_NEW_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50
    SLEEP_DURATION: float = 5.5  # seconds
    GPU_ID: Optional[int] = int(os.getenv("GPU_ID", 0))  # Default to GPU 0. Set to None to use CPU.
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")  
    expressions = {"excite","sad","hug","think","listen","grief","confused","sleep","surprise","love","dizzy","suspicious","correct","admire","worry","scold","blink","fear"}
    char_rate = 17
    img_flag = False
    imag = None

# Initialize Logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class FlashAttentionLayer(torch.nn.Module):
    def __init__(self, original_attention_module):
        super(FlashAttentionLayer, self).__init__()
        self.original_attention = original_attention_module

    def forward(self, query, key, value, attn_mask=None, *args, **kwargs):
        qkv = torch.cat([query, key, value], dim=-1)
        output = flash_attn_func(
            qkv,
            attn_mask,
            self.original_attention.num_heads,
            self.original_attention.head_dim,
            self.original_attention.dropout.p if hasattr(self.original_attention, 'dropout') else 0.0,
            return_softmax=False
        )
        return output

class MistyAssistant:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.recognizer = sr.Recognizer()

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
        self.ind = 1
        self.model = None
        self.processor = None
        self.misty = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are Misty, a social robot to interact with users and listen to their instruction. Converse with the users in a friendly tone."}
        ]

    def load_model(self):
        try:
            logger.info(f"Loading Hugging Face model: {Config.MODEL_NAME}...")
            start_time = time.time()
            os.makedirs(Config.MAIN_CACHE_DIR, exist_ok=True)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=Config.MAIN_CACHE_DIR,
                use_auth_token=Config.HUGGINGFACE_TOKEN 
            ).half().to(self.device)

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    logger.info(f"Replacing {name} with FlashAttentionLayer")
                    parent_module = self._get_parent_module(self.model, name)
                    setattr(parent_module, name.split('.')[-1], FlashAttentionLayer(module))
                elif hasattr(module, 'self_attn') and isinstance(module.self_attn, torch.nn.MultiheadAttention):
                    logger.info(f"Replacing {name}.self_attn with FlashAttentionLayer")
                    setattr(module, 'self_attn', FlashAttentionLayer(module.self_attn))

            self.processor = AutoProcessor.from_pretrained(
                Config.MODEL_NAME,
                cache_dir=Config.MAIN_CACHE_DIR,
                use_auth_token=Config.HUGGINGFACE_TOKEN  
            )
            logger.info("Model and processor loaded successfully with Flash Attention.")
            self.timings['Model loading'] = time.time() - start_time

        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

    def _get_parent_module(self, model, module_name):
        components = module_name.split('.')
        parent = model
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent

    @staticmethod
    def extract_misty_response(response: str,index) -> str:
        pattern = r'^(?:System|Misty):\s*(.*)'
        matches = re.findall(pattern, response, re.MULTILINE)
        print(response)
        if len(matches) >= 2:
            return matches[index].strip()
        else:
            return ""

    @staticmethod
    def apply_chat_template(messages: List[Dict[str, str]],img_flag) -> str:
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
        '''
        if img_flag:
            formatted_messages = "<|image|><|begin_of_text|>\n" + formatted_messages
        '''
        formatted_messages += "Misty: "
        return formatted_messages

    def generate_response_transformers(self) -> str:
        try:
            formatted_prompt = self.apply_chat_template(self.messages, img_flag=Config.img_flag)
            logger.debug("Formatted prompt for model:\n%s", formatted_prompt)
            logger.info("Generating response...")
            if Config.img_flag:
                #image = Image.open(Config.IMAGE_SAVE_PATH).convert("RGB")
                inputs = self.processor(
                    images=Config.imag,
                    text=formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_INPUT_LENGTH
                ).input_ids.to(self.device)
                Config.img_flag = False
            else:
                inputs = self.processor(
                    text=formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=Config.MAX_INPUT_LENGTH
                ).input_ids.to(self.device)                

            outputs = self.model.generate(
                inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
            )
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.debug("Raw Generated response: %s", response)
            misty_response = self.extract_misty_response(response, self.ind)
            logger.info("Extracted Misty's response: %s", misty_response)

            return misty_response
        except Exception as e:
            logger.exception(f"Failed to generate response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def download_audio_file(self, audio_url: str) -> bool:
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
            return "did not understand"
        except Exception as e:
            logger.exception(f"Unexpected error during transcription: {e}")
        return None
    
    def download_image(self):
        try:
            timeout = 10
            response = requests.get(Config.IMAGE_URL, timeout=timeout)
            response.raise_for_status()  
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.save(Config.IMAGE_SAVE_PATH)
            Config.imag = image
            print(f"Image successfully downloaded and saved as {Config.IMAGE_SAVE_PATH}")
        except requests.exceptions.Timeout:
            print(f"Error: The request timed out after {timeout} seconds.")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # e.g., 404 Not Found
        except requests.exceptions.RequestException as req_err:
            print(f"Error during request: {req_err}")  # Other request-related errors
        except IOError as io_err:
            print(f"IO error occurred while saving the image: {io_err}")  # File saving errors
        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # Any other exceptions

    def capture_speech_callback(self, data):
        logger.debug("Voice capture callback triggered.")

    def start_voice_capture(self):
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
    def perform_action(self, action:str):
        u = Config.MISTY_IP
        url = f"http://{u}/api/actions/start"
        params = {
            "name": action,
            "useVisionData": "false"}
        headers = {
            "Content-Type": "application/json"}
        data = {
            "name": action,
            "useVisionData": False}
        def timeout_error():
            raise TimeoutError("Request timed out")

        timer = Timer(10, timeout_error)
        try:
            timer.start()
            response = requests.post(url, params=params, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            print(json.dumps(json_data, indent=4))
        except TimeoutError as e:
            print(e)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        finally:
            timer.cancel()
    def sub_word(self, sub, word):
        if len(sub) > len(word):
            return False
        i = iter(word)
        return all(char in i for char in sub)

    def check_expressions(self, response:str):
        words = response.split()
        char_count = 0
        events = []
        r = []
        w = ""
        expressions = sorted(Config.expressions, key=len)
        for word in words:
            for exp in expressions:
                if self.sub_word(exp, word):
                    start_time = char_count/Config.char_rate # character rate for TTS
                    r.append(w)
                    events.append((start_time,exp))
                    break
            w += word
            char_count += len(word)+1
        if len(r) < 1:
            r.append(response)
        print("Response chunks generated:", response)
        return events , r


    def run(self):
        try:
            self.load_model()
            self.misty = Robot(Config.MISTY_IP)
            logger.info("Connected to Misty at %s", Config.MISTY_IP)
            self.messages.append({"role":"user","content":"introduce yourself as Misty"})
            response = self.generate_response_transformers()          
            self.messages.append({"role": "assistant", "content": response})
            self.misty.speak(response, None, None, None, True, "tts-content")
            duration_speaking = len(response)/Config.char_rate
            self.perform_action("yes2")
            time.sleep(duration_speaking-1)

            
            # convo will go on indefinitely
            while True:
                overall_start_time = time.time()
                self.ind += 1                    
                # Start voice capture
                start_time = time.time()
                self.start_voice_capture()

                self.timings['Misty speech capture'] = time.time() - start_time

                # Download the audio file
                start_time = time.time()
                if self.download_audio_file(Config.AUDIO_URL):
                    self.timings['Audio file download'] = time.time() - start_time
                else:
                    logger.error("Audio file download failed.")
                    self.perform_action(action="worry")
                    self.misty.speak("Pardon me, there was a glitch in my processing. Can you please repeat?",None, None, None, True, "tts-content")
                    time.sleep(68/Config.char_rate) # placeholder for above response
                    continue

                # Voice transcription using Google
                start_time = time.time()
                user_text = self.transcribe_audio()
                if user_text:
                    if user_text == "did not understand":
                        self.messages.append({"role":"user","content":""})
                    else:
                        if "photo" in  user_text:
                            pattern = r'\bphoto\b'
                            user_text.lower()
                            clean_text = re.sub(pattern, '', user_text)
                            user_text = "<|image|><|begin_of_text|>"+clean_text
                            Config.img_flag = True
                        self.messages.append({"role": "user", "content": user_text})
                    self.timings['Voice transcription'] = time.time() - start_time      
                    logger.info(f"User said: {user_text}")
                else:
                    logger.error("Voice transcription failed. Exiting.")
                    self.perform_action(action="sad")
                    self.misty.speak("Pardon me, there was a glitch in my processing. Can you please repeat?",None, None, None, True, "tts-content")
                    time.sleep(68/Config.char_rate) # placeholder for above response
                    continue
                if "satisfied with my care" in user_text:
                    logger.info("Ending conversation")
                    break
                if Config.img_flag:
                    logger.info("Capturing pic")
                    self.perform_action("body-reset")
                    im_st = time.time()
                    self.download_image()
                    Config.img_flag = True
                    im_dur = time.time() - im_st
                # Generate response using the model
                start_time = time.time()
                response = self.generate_response_transformers()
                if response == "":
                    repo = "I am sorry, I didn't quite catch that. Can you please repeat?"
                    self.misty.speak(repo, None, None, None, True, "tts-content" )
                    time.sleep(len(repo)/Config.char_rate)
                    continue
                
                self.timings['LLM execution'] = time.time() - start_time

                # Append assistant's response to messages
                self.messages.append({"role": "assistant", "content": response})

                # Misty speaks the response
                start_time = time.time()
                dur_time = time.time()
                self.perform_action(action="body-reset")
                '''
                logger.info("Misty's response: %s", response)
                st_time = time.time()
                self.misty.speak(response, None, None, None, True, "tts-content")
                if "excite" in  response:
                    self.perform_action(action="admire")
                for t, exp in events:
                    if time.time() - st_time == t:
                        self.perform_action(action=exp)
                if time.time() - st_time == 3:
                    self.perform_action("hug")
                '''
                events,r = self.check_expressions(response)
                st_time = time.time()
                self.misty.speak(response, None, None, None, True, "tts-content")
                for res,(t,exp) in zip(r,events):
                    self.perform_action(action=exp)

                #logger.info("Length of response: ", len(response))
                duration_speaking = len(response)/Config.char_rate # average TTS speaking time: 17 characters per second
                self.timings['Misty Response processing'] = time.time() - start_time

                overall_end_time = time.time()
                self.timings['Total time taken'] = overall_end_time - overall_start_time
                self.timings['Latency'] = (self.timings['Misty speech capture'] + self.timings['Audio file download'] + self.timings['Voice transcription'] + self.timings['LLM execution'])
                self.timings['Latency excluding speech capture'] = self.timings['Latency'] - self.timings['Misty speech capture']

                # Log timings
                for process, duration in self.timings.items():
                    logger.info(f"{process}: {duration:.2f} seconds")
                seconds_done = time.time() - dur_time

                time.sleep(duration_speaking-seconds_done-2)
                if self.ind > 1:
                    self.perform_action(action="listen")
            self.perform_action(action="happy")
            self.misty.speak("I am glad to have helped you out today. I hope you have a good day. Take care! Bye!", None, None, None, True, "tts-content")
            self.perform_action(action="body-reset")

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
