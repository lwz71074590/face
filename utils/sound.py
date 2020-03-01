import pyaudio
import wave


def alarming(wav_path):
    """
    播放报警音频
    :param wav_path: 音频文件所在路径
    """

    chunk = 1024
    f = wave.open(wav_path, "rb")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    data = f.readframes(chunk)

    while data != b'':
        stream.write(data)
        data = f.readframes(chunk)

    stream.stop_stream()
    stream.close()

    p.terminate()
