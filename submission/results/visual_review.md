# Post-run visual inspection

Reviewer: Codex assistant, non-expert visual inspection on 2026-10-23. All nine fixed-seed class grids were viewed: three pages each for the first-round VAE, first-round GAN and revised GAN. Each class shows four generated and four held-out real examples. This is a small descriptive inspection, not human histopathology review, a calibrated score, or a diagnosis of training failure. It did not affect model selection, training or numerical recommendations.

The automatic packets' original `not_yet_reviewed` fields remain historical records. This appendix documents the subsequent assistant inspection separately; it does not assert expert approval or representative coverage beyond the displayed samples. Sample indices are printed in the grids and saved in their visual_review.json files.

| Class | VAE round 1 | GAN round 1 | GAN round 2 |
|---|---|---|---|
| 0 | Strong pixel noise; large clear spaces and curved boundaries seen in real examples are absent. | Diffuse pink/purple mottling; fewer clear spaces and boundaries. | Pale mottling with visible pixel/checker patterns; lacks the large clear spaces. |
| 1 | Noisy multicolor pixels unlike the smoother real fields. | Some dark/pale variation, but mottled relative to real examples. | Four similarly mottled fields; narrower dark/light variation than displayed real examples. |
| 2 | Pixel noise obscures the distinct light/dark structures seen in real examples. | Soft pink/purple mottling with less distinct structure. | Coarse dark patches and repeated pixel patterns; fewer distinct clear regions. |
| 3 | Noisy pixels replace the real examples' more distinct light/dark elements. | Softer mottling with some contrasting patches. | Mostly similar purple fields with pixel patterns; limited visible variation. |
| 4 | Noisy fields lack the elongated orientations in real examples. | Mottling lacks the clear directional bands of real examples. | Faint vertical/pixel patterns, without comparable organized bands and spaces. |
| 5 | Noise dominates; large pale spaces and bands are missing. | Mostly smooth pink/mottled fields, with some contrast variation. | Mottled fields with repeated spots; missing the pronounced spaces/bands in some real examples. |
| 6 | Noise obscures the real examples' curved boundaries and openings. | Contrast varies but boundaries are diffuse. | Variable brightness and coarse patches; few comparable curved boundaries/openings. |
| 7 | Noisy fields lack the elongated structures present in some real examples. | Soft mottling and color variation, without comparable elongated detail. | Similar mottled fields; directional detail and between-sample variation appear limited. |
| 8 | Pixel noise; large clear spaces and organized edges are missing. | Mottling with fewer clear spaces and edges. | Repeated coarse patterns; lacks prominent clear spaces in several real examples. |

These observations support investigating spatial structure and noise, but do not prove the cause or quantify quality over the full generated datasets. Do not treat an apparently smoother image as evidence of downstream usefulness.

## Inspected packets

- VAE round 1: [classes 0–2](pathmnist_recovery_v1/round_1/evaluation/image_vae/visual_samples_1.png), [classes 3–5](pathmnist_recovery_v1/round_1/evaluation/image_vae/visual_samples_2.png), [classes 6–8](pathmnist_recovery_v1/round_1/evaluation/image_vae/visual_samples_3.png).
- GAN round 1: [classes 0–2](pathmnist_recovery_v1/round_1/evaluation/image_gan/visual_samples_1.png), [classes 3–5](pathmnist_recovery_v1/round_1/evaluation/image_gan/visual_samples_2.png), [classes 6–8](pathmnist_recovery_v1/round_1/evaluation/image_gan/visual_samples_3.png).
- GAN round 2: [classes 0–2](pathmnist_continuation_v1/evaluation/image_gan/visual_samples_1.png), [classes 3–5](pathmnist_continuation_v1/evaluation/image_gan/visual_samples_2.png), [classes 6–8](pathmnist_continuation_v1/evaluation/image_gan/visual_samples_3.png).
