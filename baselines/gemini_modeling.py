import os
import time
import random
random.seed(43)
import numpy as np
import google.generativeai as genai

from gemini.extract_frames import extract_frame_from_video
from gemini.upload import File, get_timestamp, make_request

from base import ViLLMBaseModel

# GOOGLE_API_KEY_POOL = ["AIzaSyBPnXWH0qFGb8Z_JWyRieMGex-uMqmAYr0", "AIzaSyA7tyrZIhCwUWF42ImL2Uf3Xo17tlmEGNI", "AIzaSyCyido7T6ps2ID-N0xGRnOo0icX54a9tz0", "AIzaSyDo8kTKe0tfzeYgpPA6al47z8Lx5vkh74s", "AIzaSyBxazB94KwCPE8-oH3_5b91RWTupF0_8H0", "AIzaSyDH-CjgRIMfr2a65VKccHinhHmn1uZEIII"]
from os import getenv
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY_POOL = eval(getenv("GEMINI"))

from google.generativeai.types import HarmCategory, HarmBlockThreshold
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
}


class Gemini(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )

        # self.frame_extraction_directory = model_args["frame_path"]\
        self.frame_extraction_directory = "./cache/gemini15"
        self.frame_prefix = "_frame"

    def generate(self, instruction, video_path):
        # assert len(videos) == 1
        # video_path = videos[0]
        # instruction = instruction[0] if type(instruction)==list else instruction

        # 1. extract frame: default 1 fps
        # get extract frame 
        vid = video_path.split("/")[-1]
        name, ext = os.path.splitext(vid)
        self.frame_extraction_files = os.path.join(self.frame_extraction_directory, name)
        if not os.path.exists(self.frame_extraction_files):
            extract_frame_from_video(video_path, self.frame_extraction_files, self.frame_prefix)

        # 2. upload frame to gemini
        files = os.listdir(self.frame_extraction_files)
        files = sorted(files)
        files_to_upload = []
        max_frame_length = 128
        if len(files) > max_frame_length:
            sample_idx = np.linspace(0, len(files)-1, max_frame_length, dtype=int).tolist() # FIXME: the API is easily broken up
            for idx in sample_idx:
                files_to_upload.append(
                    File(file_path=os.path.join(self.frame_extraction_files, files[idx]), frame_prefix=self.frame_prefix))
        else:
            for file in files:
                files_to_upload.append(
                    File(file_path=os.path.join(self.frame_extraction_files, file), frame_prefix=self.frame_prefix))

        # for file in files:
        #         files_to_upload.append(
        #             File(file_path=os.path.join(self.frame_extraction_directory, file), frame_prefix=self.frame_prefix))



        while 1:
            try:
                GOOGLE_API_KEY = random.choice(GOOGLE_API_KEY_POOL)
                genai.configure(api_key=GOOGLE_API_KEY)
                # Upload the files to the API
                # Only upload a 10 second slice of files to reduce upload time.
                # Change full_video to True to upload the whole video.
                full_video = True
                uploaded_files = []
                # print(f'Uploading {len(files_to_upload) if full_video else 10} files. This might take a bit...')
                for file in files_to_upload if full_video else files_to_upload[:10]:
                    # print(f'Uploading: {file.file_path}...')
                    response = genai.upload_file(path=file.file_path)
                    file.set_file_response(response)
                    uploaded_files.append(file)
                # print(f"Completed file uploads!\n\nUploaded: {len(uploaded_files)} files")

                # 3. generate
                # Create the prompt.
                prompt = instruction
                # Set the model to Gemini 1.5 Pro.
                model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
                request = make_request(prompt, uploaded_files)
                # print(request)
                response = model.generate_content(request,
                                                request_options={"timeout": 600}, safety_settings=SAFETY_SETTINGS)

                # delete uploaded files to save quota
                # print(f'Deleting {len(uploaded_files)} images. This might take a bit...')
                for file in uploaded_files:
                    genai.delete_file(file.response.name)
                    # print(f'Deleted {file.file_path} at URI {file.response.uri}')
                # print(f"Completed deleting files!\n\nDeleted: {len(uploaded_files)} files")
                response = response.text
                time.sleep(random.randint(5, 10))
                break
            except Exception as e:
                print(e)
                if "blocked" in str(e):
                    response = ""
                    break
                time.sleep(random.randint(0, 10))
        return response