import torch
import warnings
import onnxruntime
import numpy as np
import os
warnings.filterwarnings("ignore")

from speechbraindev.pretrained import SpectralMaskEnhancement

def to_numpy(tensor):
    output = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return output

if __name__ == "__main__":
    print("Welcome to TonSpeech")
    print("Creating MetricGAN+ model")
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )
    print("Loading dummy input")
    dummy_input = enhance_model.load_audio("data/dummy_input.wav")
    dummy_input = dummy_input.unsqueeze(0)
    torch_out = enhance_model(dummy_input)
    os.remove("dummy_input.wav")
    # print(dummy_input.shape)
    print('Exporting to ONNX Model')
    torch.onnx.export(enhance_model, 
                      dummy_input, 
                      "metricganp.onnx",
                      export_params=True,
                      opset_version=10,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    print("Model has been exported to ONNX!")

    # |--------------------------------------------------------------------------------|
    # |verify that the exported model computes the same values when run in ONNX Runtime|
    # |--------------------------------------------------------------------------------|
    print("--------------------------------")
    print("Verifying the exported model...")
    try:
        ort_session = onnxruntime.InferenceSession("metricganp.onnx")
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    except:
        print("Something went wrong :'(")