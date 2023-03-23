# Import the required module for text 
# to speech conversion
from gtts import gTTS
from IPython.display import Audio
  
# This module is imported so that we can 
# play the converted audio
import os
  
# The text that you want to convert to audio
mytext = 'Welcome to geeksforgeeks!'
  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("welcome.mp3")
sound_file = 'welcome.mp3'

#playing audio
Audio(sound_file, autoplay=True)
  
# Playing the converted file
#os.system("mpg321 welcome.mp3")

from IPython.display import Audio
from gtts import gTTS
tts = gTTS('hello joyjit')
tts.save('1.wav')
sound_file = '1.wav'
Audio(sound_file, autoplay=True)