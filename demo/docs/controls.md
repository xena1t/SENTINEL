# Demo control reference

## Recorded inputs

The demo contains one drone example and one background example per dataset.
The [data verification command](maintenance.md#verify-the-evidence-pack) regenerates the example counts.

| Dataset | Camera | RF | Audio | Playback |
| --- | --- | --- | --- | --- |
| TRIDENT test | Video | Spectrograms | Paired recording | Video with overlapping windows |
| RFVS external | Still image | 30 ms capture | Not recorded | One fixed observation |
| Halmstad external | Video | Not recorded | Not recorded | Video with nonoverlapping windows |

## Controls

| Control | Behavior |
| --- | --- |
| Dataset **YES** and **NO** buttons | Select the drone or background example and hold its saved cue. Restore all recorded inputs. |
| Example dropdown | Select the same examples as the dataset buttons. |
| Camera, RF, and audio switches | Hold the video window and select the cached scores for that input combination. |
| **Play recording** | Play the video and advance the displayed scores as windows finish. |
| **Hold window** | Pause and hold the latest completed video window. |
| Previous and next arrows | Hold the previous or next cached video window. |
| Timeline slider | Select and hold a cached video window. |
| **Listen** | Enable or mute audible TRIDENT audio. This does not change model inputs. |
| Fullscreen icon | Enter or exit fullscreen. |
| Released sensor combinations table | List the supported combinations and their exact saved scores for the selected observation. |
| **Source & model provenance** | Show dataset labels, source identifiers, and evaluation details. |
| **Follow the evidence through the model** | Expand the pipeline explanation. |

Unrecorded inputs stay disabled. Excluded inputs were recorded but are switched off for the model comparison.
With every recorded input switched off, the demo displays no prediction.
RFVS has no playback, seek, or window-step controls because each example is one fixed observation.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| **Space** | Play or pause video. |
| **Left arrow** or **Right arrow** | Select the previous or next video window. |
| **1** | Toggle camera evidence when recorded. |
| **2** | Toggle RF evidence when recorded. |
| **3** | Toggle audio evidence when recorded. |
| **F** | Enter or exit fullscreen. |

Focused form controls retain their native keyboard behavior.

## Model cards

Both cards use the same observation and selected inputs.
A drone score of at least `0.5` produces a drone decision.
The displayed score has three decimal places. Its tooltip preserves the stored value.
The change below each card compares the selected inputs with all inputs recorded for that example.
The timeline shows saved scores without interpolation.

The **DATASET LABEL** is the source label. A model decision can disagree with that label.
The [evidence reference](evidence.md) defines timing, probabilities, and source limitations.
