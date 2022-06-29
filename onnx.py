import torch
import torchaudio
import warnings
import onnxruntime
warnings.filterwarnings("ignore")

from speechbraindev.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
        source="pretrained_models/metricgan-plus-voicebank"
    )

dummy_input = enhance_model.load_audio("data/voice-bank/voice-bank-slr/clean_testset_wav/p232_001.wav")
dummy_input = dummy_input.unsqueeze(0)
print("Dummy input shape: ", dummy_input.shape)

torch.onnx.export(enhance_model, 
                  dummy_input, 
                  "metricganplus.onnx",
                  opset_version=15)
print("Successful!!!")