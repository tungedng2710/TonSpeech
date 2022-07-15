import torch
import warnings
import onnxruntime
import numpy as np
import os
import torchaudio
import argparse

from src.speechbraindev.pretrained import SpectralMaskEnhancement
from src.cmgan_onnx import CMGAN_ONNX

warnings.filterwarnings("ignore")

def verify(onnx_model_path,
           torch_model,
           dummy_input):
    assert onnx_model_path == None
    assert torch_model == None
    assert dummy_input == None
    print("Verifying the exported model...")
    torch_out = torch_model(dummy_input)
    try:
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    except:
        print("Something went wrong :'(")

def to_numpy(tensor):
    output = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return output

def metricganp_to_onnx():
    print("Creating MetricGAN+ model...")
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )
    print("Loading dummy input...")
    dummy_input = enhance_model.load_audio("data/dummy_input.wav")
    dummy_input = dummy_input.unsqueeze(0)
    os.remove("dummy_input.wav")
    print('Exporting to ONNX Model...')
    torch.onnx.export(enhance_model, 
                      dummy_input, 
                      "metricganp.onnx",
                      export_params=True,
                      opset_version=15,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    print("Model has been exported to ONNX!")
    verify("metricganp.onnx", enhance_model, dummy_input)

def cmgan_to_onnx():
    checkpoint_path = "./pretrained_models/CMGAN/cmgan_ckpt"
    dummy_input_path = "./data/noisy_sample_16k.wav"
    print("Creating Conformer-based GAN model...")
    cmgan_onnx_model = CMGAN_ONNX(checkpoint_path=checkpoint_path,
                                  device_id=0)
    print("Loading dummy input...")
    dummy_input, sr = torchaudio.load(dummy_input_path)
    assert sr == 16000
    # torch_out = cmgan_onnx_model(dummy_input)

    print('Exporting to ONNX Model...')
    torch.onnx.export(cmgan_onnx_model, 
                      dummy_input, 
                      "cmgan.onnx",
                      export_params=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    print("Model has been exported to ONNX!")

if __name__ == "__main__":
    print("+------------------------------+")
    print("|           TonSpeech          |")
    print("+------------------------------+")

    cmgan_to_onnx()