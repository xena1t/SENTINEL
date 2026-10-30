# Present the three dataset pairs

Start the demo with the [startup guide](../README.md), then use this sequence to explain the comparison.

## Show the TRIDENT test pair

1. Select **YES** under **TRIDENT**. Point out the camera video, RF spectrogram, and paired audio waveform.
2. Read the dataset label and both model decisions for the selected window.
3. Select **Play recording**, then **Hold window** at a moment you want to compare.
4. Switch off one input. Compare the saved scores while the window stays fixed.
5. Restore the input. Try another input combination.
6. Select **Listen** to hear the paired recording. Use the audio input switch separately to include or exclude audio from the models.
7. Select **NO** under **TRIDENT** and repeat the comparison on the background recording.

## Show the RFVS external pair

1. Select **YES** under **RFVS**. Point out the camera still and the RF spectrogram.
2. Switch between camera alone, RF alone, and both inputs.
3. Explain that the example contains one still and a 30 ms RF capture. There is no video to play or score history to seek.
4. Select **NO** under **RFVS**. Leave both recorded inputs enabled.
5. Compare the decisions. Fusion classifier produces a false alarm, while Qwen detector rejects this background example.

## Show the Halmstad external pair

1. Select **YES** under **Halmstad** and play the drone video.
2. Point out that RF and audio are marked **NOT RECORDED**. The model comparison uses camera evidence only.
3. Select **NO** under **Halmstad**. Leave the selected 5 to 6 second window held.
4. Compare the decisions. Fusion classifier produces a false alarm on the bird, while Qwen detector rejects that window.
5. Play or step through the other windows to show how the saved scores vary over the recording.

## Explain the pipeline

1. Expand **Follow the evidence through the model**.
2. Follow the encoder, fusion, classifier, connector, and Qwen stages.
3. On TRIDENT, toggle an input and point out the corresponding illustrated route.
4. State that the routes illustrate the architecture. They do not show measured attention.
5. Expand **Source & model provenance** to identify the recording and saved evaluation.

Describe the demo as a replay of real recordings and saved Version 2 predictions.
Do not describe playback as live inference or recording time as GPU latency.
Present these selected examples as illustrations, rather than a whole-dataset accuracy comparison.
