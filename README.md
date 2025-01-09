# TheraMisty
TheraMisty is a cutting-edge social robot equipped with multimodal capabilities, designed to provide general therapy and emotional support to humans. Leveraging advanced speech recognition, natural language processing, and visual recognition technologies, TheraMisty offers a safe and engaging space for users to talk, listen, and share their feelings.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Operating TheraMisty](#operating-theramisty)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)

## Features

- **Conversational AI:** Utilizes Google's speech recognition and Meta's LLaMA language models to maintain natural, human-like conversations with memory capabilities.
- **Emotion Expression:** Capable of expressing a range of emotions including happiness, sadness, anger, worry, concern, and thoughtfulness to enhance user engagement.
- **Visual Recognition:** Detects and interprets users' micro-expressions to generate contextually appropriate responses.
- **Low Latency:** Maintains an average response latency of 1.74 seconds, ensuring smooth and timely interactions.
- **Data Privacy:** Processes all operations on a commodity server, ensuring full data privacy and security.
- **Expandable:** Easily integrates additional functionalities and can be customized to suit various therapeutic needs.

## Current application: 
### Speech Language Therapy 
- Provided to children with special needs (autism, dyslexia) with accurate robotic function calling.
- Employed three Agentic AIs to work in series to develop synthetic data on therapy scene, session planning and therapist conversation alignment. 
- Finetuned LLaMA language models using PEFT-LoRA configuration, Distributed Data Parallelism, Batchwise Lazy Loading integrated with 4-bit quantization at FP16 across 4 NVIDIA RTX 3090 GPUs.
- Performed zero shot evaluation for finetuned models with speech language pathologists at National AI Institute and with GPT-4o. 

## Demo
<div align="center">
  <img src="misty_intro_git.gif" alt="Misty Introduction GIF" width="600" />
</div>
Misty communicating without expressions: <a href="https://www.youtube.com/watch?v=ZXgYyf2mxcU" target="_blank">
  <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white" alt="YouTube">
</a>
Misty communicating with expressions: <a href="https://youtube.com/shorts/bM_8sR366X4" target="_blank">
  <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white" alt="YouTube">
</a>

## Installation

### Prerequisites

Before installing TheraMisty, ensure that your system meets the following requirements:

- **Operating System:** Linux-based OS (e.g., Ubuntu 20.04+)
- **Python:** Version 3.8 or higher
- **Hardware:** A compatible robot platform with network capabilities
- **GPU:** NVIDIA GPU with CUDA support (optional, for improved performance)
- **Internet Connection:** Required for downloading models and speech recognition services

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/TheraMisty.git
   cd TheraMisty
2. **Create a Virtual Environment**
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. **Install Required Libraries**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
4. requirements.txt file should include: 
   ```bash
   torch
   transformers
   speechrecognition
   requests
   mistyPy
   flash-attn

5. **Configure Environment Variables**
   Create a .env file in the project root directory and add the following configurations:
   ```bash
   GPU_ID=0
   HUGGINGFACE_TOKEN=your_huggingface_token
   MISTY_IP=your_misty_robot_ip

6. **Connect to Misty**
   Ensure that your Misty robot is connected to the same network and accessible via the IP address specified in the .env file.

7. **Run code**
   ```bash
   python connect_llama_flash_attn.py
---

## Configuration

The `Config` class within the `TheraMisty.py` script centralizes all configurable parameters, allowing for easy customization and management of the robot's behavior and environment. Below is a detailed breakdown of each configuration parameter:

### Directories

- **`MAIN_CACHE_DIR`**
  - **Type:** `str`
  - **Description:** Specifies the main directory for caching models. This is where pre-trained models are stored to avoid repeated downloads.
  - **Default Value:** `"/home/user/misty/Huggingface/"`

- **`MODEL_NAME`**
  - **Type:** `str`
  - **Description:** Defines the name of the language model to be used by TheraMisty.
  - **Default Value:** `"meta-llama/Llama-3.2-1B-Instruct"`

- **`LOCAL_MODEL_DIR`**
  - **Type:** `str`
  - **Description:** Derives the local directory path for the specified model by combining `MAIN_CACHE_DIR` and `MODEL_NAME`.
  - **Default Value:** `os.path.join(MAIN_CACHE_DIR, MODEL_NAME)`

- **`AUDIO_SAVE_PATH`**
  - **Type:** `str`
  - **Description:** Path where captured audio files from user interactions are saved.
  - **Default Value:** `"/home/user/misty/Python-SDK/mistyPy/misty_user_recording.wav"`

### Robot Configuration

- **`MISTY_IP`**
  - **Type:** `str`
  - **Description:** IP address of the Misty robot, enabling network communication and control.
  - **Default Value:** `"67.20.198.10"`

### API Endpoints

- **`AUDIO_URL`**
  - **Type:** `str`
  - **Description:** URL endpoint to access captured audio files from the robot.
  - **Default Value:** `f"http://{MISTY_IP}/api/audio?fileName=capture_Dialogue.wav&base64=false"`

### Logging

- **`LOG_LEVEL`**
  - **Type:** `int`
  - **Description:** Sets the logging level for the application (e.g., INFO, DEBUG).
  - **Default Value:** `logging.INFO`

- **`LOG_FORMAT`**
  - **Type:** `str`
  - **Description:** Defines the format for logging messages, including timestamp, logger name, log level, and message.
  - **Default Value:** `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`

### Model Parameters

- **`MAX_INPUT_LENGTH`**
  - **Type:** `int`
  - **Description:** Maximum number of tokens allowed for the model's input.
  - **Default Value:** `2048`

- **`MAX_NEW_TOKENS`**
  - **Type:** `int`
  - **Description:** Maximum number of tokens the model can generate in response.
  - **Default Value:** `500`

- **`TEMPERATURE`**
  - **Type:** `float`
  - **Description:** Sampling temperature for response generation; higher values lead to more randomness.
  - **Default Value:** `0.7`

- **`TOP_P`**
  - **Type:** `float`
  - **Description:** Nucleus sampling parameter that controls the cumulative probability for token selection.
  - **Default Value:** `0.9`

- **`TOP_K`**
  - **Type:** `int`
  - **Description:** Limits the number of highest probability vocabulary tokens to keep for top-k filtering.
  - **Default Value:** `50`

### Performance

- **`SLEEP_DURATION`**
  - **Type:** `float`
  - **Description:** Duration in seconds that the system sleeps between actions to manage timing.
  - **Default Value:** `5.5`

- **`GPU_ID`**
  - **Type:** `Optional[int]`
  - **Description:** Identifier for the GPU to be used for model processing. Set to `None` to use the CPU.
  - **Default Value:** `int(os.getenv("GPU_ID", 0))`

- **`HUGGINGFACE_TOKEN`**
  - **Type:** `Optional[str]`
  - **Description:** API token for accessing Hugging Face models, especially private or restricted ones.
  - **Default Value:** `os.getenv("HUGGINGFACE_TOKEN")`

### Expressions and Rates

- **`expressions`**
  - **Type:** `set`
  - **Description:** Set of emotions that TheraMisty can express to engage users effectively.
  - **Default Value:** `{"excite", "sad", "hug", "think", "listen"}`

- **`char_rate`**
  - **Type:** `int`
  - **Description:** Character rate used for synchronizing text-to-speech (TTS) timing.
  - **Default Value:** `17`

**Ensure these configurations align with your environment and requirements.**

---

## Technologies Used

TheraMisty integrates a suite of advanced technologies and libraries to deliver seamless and intelligent interactions. Below is an overview of the key technologies employed in the project:

### **Hardware and Robotics**

- **[Misty Robotics](https://www.mistyrobotics.com/)**
  - **Purpose:** Provides the platform and SDK (`mistyPy`) for controlling and interfacing with the Misty robot. Enables network communication, event handling, and action execution.

### **Additional Libraries**

- **[mistyPy](https://github.com/mistyrobotics/mistyPy)**
  - **Purpose:** Python SDK specifically designed for interacting with Misty robots, enabling control over movements, speech, and other robotic functions.

---

*By leveraging these technologies, TheraMisty achieves a harmonious blend of robotics, natural language processing, and real-time interaction capabilities, providing users with an engaging and supportive therapeutic experience.*

---

