""" Create StreamLit model app for project
1. Load model
2. Load whisper ai api
3. Load example audio file
4. Create transcript from audio
5. Predict from model
6. Allow audio record
"""

from typing import Tuple, List, Dict, Any
import pickle
import torch
import torch.nn as nn
import torchaudio
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import streamlit as st
from st_audiorec import st_audiorec




## Define classes and functions
class FusionModel(nn.Module):

  def __init__(self):
    super(FusionModel, self).__init__()

    # Define additional layers for fusion and classification
    self.fc1 = nn.Linear(1536, 128)  # Adjust input size based on your embeddings
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 7)  # 7 output class logits for emotion classification

  def forward(self, x):
    # Pass through fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    output = self.fc2(x)
    return output

# Define a preprocessor class

class Preprocessor():

    ### To-dos: Read models, and set main preprocessing function.

    def __init__(
            self,
            audio_encoder:Tuple, text_encoder:Tuple, transcriber:whisper.load_model,
            audio_path:str, text=None, transcribe=True):
        self.audio_path = audio_path
        self.text = text
        self.transcribe = transcribe

        # Models
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.transcriber = transcriber

        # Attributes to be set later
        self.waveform = None
        self.sample_rate = None
        self.features = None


    def _read_audio(self):
        # Load audio file using torchaudio
        waveform, sample_rate = torchaudio.load(self.audio_path)
        self.waveform = waveform
        self.sample_rate = sample_rate
    
    def _transcribe_audio(self):
        # Transcribe audio file using whisper
        text = self.transcriber.transcribe(self.audio_path)["text"]
        self.text = text


    # Define a function to extract features from the waveform with a given sample rate
    def _extract_speech_features(
            self, max_length:int) -> torch.tensor :

        """Resample the data if necessary, process it and extract features from wav2vec2"""
        processor, speech_model = self.audio_encoder

        # Check if the waveform has more than one channel (stereo), if so, convert it to mono by averaging the channels
        if self.waveform.ndim == 2:
            self.waveform = self.waveform.mean(dim=0)

        # If the sample rate of the waveform is not 16000 Hz, resample it to 16000 Hz
        if self.sample_rate != 16000:

            # Create a resampler object to convert the waveform's sample rate to 16000 Hz
            resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)

            # Apply the resampler to the waveform
            self.waveform = resampler(self.waveform)

        # Process the waveform to prepare it for the Wav2Vec2 model, converting it to tensors and padding if necessary
        wav2vec2_features = processor(self.waveform, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=max_length)
        with torch.no_grad():
                wav2vec2_features = speech_model(**wav2vec2_features)
        wav2vec2_features = wav2vec2_features.last_hidden_state.mean(dim=1)
        wav2vec2_features = nn.functional.normalize(wav2vec2_features, dim=1).squeeze() # Squeeze to get 768 dim vector

        # Return wav2vec2_features
        return wav2vec2_features
    

    def _extract_text_features(self, max_length:int) -> torch.tensor:

        """
        Tokenize text and extract embeddings from BERT Encoder component of tinyBERT model.
        Then the features are passed into PCA for dimension reduction.
        """
        tokenizer, text_encoder = self.text_encoder
        text = self.text
        # Tokenize text
        tokenized_text = tokenizer(text, padding="max_length", max_length=max_length, return_tensors="pt")

        # Feed into BERT Encoder
        # Disable gradient calculation
        with torch.no_grad():
            # Forward pass up to the BertEncoder
            text_features = text_encoder(**tokenized_text, output_hidden_states=True)
            text_features = text_features.hidden_states[-1].squeeze()

        # Normalize and reduce because the dimension count is huge
        text_features = nn.functional.normalize(text_features)

        # Use PCA for dimensionality reduction on text tensor to match dimension of speech tensor
        pca = PCA(n_components=6)
        text_features = torch.tensor(pca.fit_transform(text_features).flatten())

        # Return the list of features
        return text_features
    
    def _concat_features(self, speech_features:torch.tensor, text_features:torch.tensor) -> torch.tensor:
        """
        Concatenate the speech and text features into a single tensor.
        """

        # Concatenate the speech and text features
        features = torch.concatenate((speech_features, text_features))

        return features
    
    def preprocess(self, max_length:int=1000):
        """
        Preprocesses the audio and text data to extract features.
        """

        # Read audio file
        self._read_audio()

        # Transcribe audio file if text is not provided
        if self.transcribe:
            self._transcribe_audio()
            st.write(f"Transcribed text: {self.text}")

        # Extract speech features
        speech_features = self._extract_speech_features(max_length)

        # Extract text features
        text_features = self._extract_text_features(max_length=128)

        # Concatenate speech and text features
        features = self._concat_features(speech_features, text_features)

        return features

# Define a predict function for the model

def predict(model, features, device):
  """
  Predicts the emotion label for a given input feature.

  Args:
      model: The trained FusionModel.
      features: A tensor of shape (num_features,) representing the input features.
      device: The device to run the model on.

  Returns:
      The predicted emotion label as a string.
  """

  model.eval()  # Set model to evaluation mode
  with torch.no_grad():
    features = features.unsqueeze(0).float().to(device)  # Add batch dimension and move to device
    outputs = model(features)
    _, predicted_idx = torch.max(outputs.data, 1)
    predicted_idx = predicted_idx.item()

  # Map predicted index to emotion label
  label_mapping = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}
  predicted_label = label_mapping[predicted_idx]

  return predicted_label

def prediction_pipeline(audio_path:str, text:str, speech_encoder:Tuple, text_encoder:Tuple, whisper_model:whisper.load_model, transcribe:bool=True):
    """
    Pipeline to preprocess audio and text data and predict the emotion label.
    """

    # Preprocess audio and text data
    preprocessor = Preprocessor(audio_encoder=speech_encoder, text_encoder=text_encoder, transcriber=whisper_model, audio_path=audio_path, text=text, transcribe=transcribe)
    features = preprocessor.preprocess()

    # Predict emotion label
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predicted_label = predict(model, features, device)

    return predicted_label

############################################################################
# Load model
model = FusionModel()

# Load model weights
weights_path = "model/early_fusion.pth"
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Load text model
# load model tinyBERT tokenizer
checkpoint = "AdamCodd/tinybert-emotion-balanced"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# load tinyBERT model

with open('model/tinybert.pkl', 'rb') as f:
  tinybert = pickle.load(f)

text_encoder = (tokenizer, tinybert)

# Load speech model
speech_checkpoint = "facebook/wav2vec2-base"
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(speech_checkpoint)
wav2vec2_model = Wav2Vec2Model.from_pretrained(speech_checkpoint)

speech_encoder = (wav2vec2_processor, wav2vec2_model)

# Load transcription model
whisper_model = whisper.load_model("tiny")

############################################################################

# Load example audio file and display as playable audio in streamlit
audio_path = "data/dia54_utt3.wav"
audio_file = open(audio_path, "rb")
audio_bytes = audio_file.read()




# Create a Streamlit app
st.title("Early Fusion: Multimodal Speech Emotion Recognition")
st.header("Predict Emotion from Sample Audio File")
st.audio(audio_bytes, format="audio/wav")

# Add a button to predict the emotion from sample audio file
if st.button("Predict Emotion", key="predict_from_sample"):
    predicted_label = prediction_pipeline(audio_path, "I am happy", speech_encoder, text_encoder, whisper_model, transcribe=True)
    st.write(f"Predicted emotion: {predicted_label}")

st.header("Record and test your own audio file!")
wav_audio_data = st_audiorec()
if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    # Add button to predict emotion
    if st.button("Predict Emotion", key="predict_from_recording"):
        # Save audio data to a file
        save_path = "data/recording.wav"
        with open(save_path, "wb") as f:
            f.write(wav_audio_data)
        predicted_label = prediction_pipeline(save_path, "I am happy", speech_encoder, text_encoder, whisper_model, transcribe=True)
        st.write(f"Predicted emotion: {predicted_label}")
