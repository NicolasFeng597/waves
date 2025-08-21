"""
Live input module. 

This module contains the LiveInput class for live audio input.
"""

import pyaudio
import numpy as np

class LiveInput:
    """
    A class representing live audio input.
    """

    audio: pyaudio.PyAudio
    device_index: int

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        print("\n---------------------- record device list ----------------------")
        for i in range(numdevices):
            if self.audio.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels") > 0:
                print(
                    f"Input Device {i}"
                    + f" - {self.audio.get_device_info_by_host_api_device_index(0, i).get("name")}"
                )
        print("----------------------------------------------------------------\n")

        self.device_index = int(input("Choose a device: "))

    def get_audio(self):
        pass