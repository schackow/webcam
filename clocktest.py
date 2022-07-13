from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_wav('clock.wav')
while True:
    play(sound)
    