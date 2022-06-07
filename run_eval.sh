python eval.py \
    --eval_on_dataset 1 \
    --metric pesq \
    --clean exp/test_audio/audio_clean.mp3 \
    --denoised exp/test_audio/denoised_magicmic \
    --to_csv 0 \
    --verbose 1
# Stereo audio will be converted to mono audio automatically
# eval_on_dataset: 0 if performing on single sample and 1 for dataset
# metric: pesq or stoi
# clean: path to clean (ref) audio
# denoised: path to denoised (deg) audio
# to_csv: 1: save the result to csv file
# verbose: 1: show the progress bar, 0: show nothing