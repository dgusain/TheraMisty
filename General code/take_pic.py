import requests
from PIL import Image
import io
import json

def download_image(url: str, save_path: str, timeout: int = 10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        print(f"Image successfully downloaded and saved as {save_path}")

    except requests.exceptions.Timeout:
        print(f"Error: The request timed out after {timeout} seconds.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  
    except requests.exceptions.RequestException as req_err:
        print(f"Error during request: {req_err}")  
    except IOError as io_err:
        print(f"IO error occurred while saving the image: {io_err}")  
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 


def start_keyphrase_capture(url: str, timeout: int = 1000) -> None:
    try:
        data = {
            "overwriteExisting": None,
            "silenceTimeout": None,
            "maxSpeechLength": None,
            "captureSpeech": None,
            "speechRecognitionGrammar": None
        }
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        json_data = response.json()
        print("Response JSON Data:")
        print(json.dumps(json_data, indent=4))
        print(json_data['status'])
        #print(response)
        
    except requests.exceptions.Timeout:
        print(f"Error: The request timed out after {timeout} seconds.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
    except json.JSONDecodeError:
        print("Error: Failed to parse the response as JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    url = "67.20.193.16"
    keyphrase_url = f"http://{url}/api/audio/keyphrase/start"
    take_pic_url = f"http://{url}/api/cameras/rgb?base64=false&fileName=user_pic&displayOnScreen=false&overwriteExisting=false"

    # capture image
    #save_path = "user_pic.jpg"
    #download_image(url, save_path)
    start_keyphrase_capture(keyphrase_url)

if __name__ == "__main__":
    main()


