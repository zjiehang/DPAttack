class LoadUtil:
    @staticmethod
    def load_mel_spectrogram(file_path, n_mels=128, fmax=8000):
        import librosa

        audio, sampling_rate = librosa.load_parser(file_path)
        return librosa.feature.melspectrogram(
            y=audio, sr=sampling_rate, n_mels=n_mels, fmax=fmax)