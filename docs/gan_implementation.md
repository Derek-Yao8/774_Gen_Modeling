# Image GAN implementation

The agent now has two executable image model families: VAE and GAN. The image
GAN can be selected alone or alongside the VAE, with conditional labels for
augmentation. PBMC3K still uses the count VAE; the image GAN does not model gene
counts. Standalone planar, radial, and RealNVP expression flows are now implemented separately (see normalizing_flows_implementation.md). Diffusion remains unimplemented.

## Decisions and integration

`planning.py` advertises the GAN's executable versioned contract and validates
its implicit likelihood and `scale_pixels_minus1_1` preprocessing. Representation
learning is rejected because this GAN has no encoder. The same hyperparameter
fields control latent dimension, hidden width, batch size, epochs, learning rate,
early-stopping patience, time allowance and number of generated samples. A plan
can choose both image families within its total budget; the executor trains them
sequentially. Existing VAE implementations and their likelihoods are preserved.

The single-command pipeline requires no new flags. It can select or revise to
the GAN when the AI judges it appropriate. Feedback includes generator and
discriminator losses and the validation checkpoint score. A forced GAN objective
was used only for capability verification; ordinary planner decisions remain
objective- and data-dependent.

## Model and optimization

- `models.py`: `ImageGAN` has separate generator/discriminator modules. The
  generator projects a normal latent and optional one-hot label through linear
  layers, then two transposed convolutions and tanh. The discriminator uses two
  strided convolutions and linear layers, with spatial one-hot label channels.
  Width derives from `hidden_dim`; neither network uses batch normalization.
- `training_data.py`: the existing verified MedMNIST loader preserves official
  splits and converts GAN training pixels to [-1,1]. VAE scaling remains [0,1].
  Supported image layouts are uint8, one/three channels, spatial dimensions
  divisible by four and no larger than 64.
- `gan_trainer.py`: one discriminator update followed by one generator update
  per batch. Discriminator BCE-with-logits averages real-positive and
  detached-fake-negative losses. Non-saturating generator BCE targets real-positive
  scores while discriminator parameters are frozen. Both optimizers are Adam
  with the planned learning rate and betas (0.5,0.999); gradient norm is clipped
  at 10. Fake training labels match the real batch, avoiding a label-frequency
  shortcut for the discriminator.
- Device probing exercises forward/backward operations. Existing execution
  supervision enforces the monitored memory/time limits; training reserves 20%
  of its time allowance for final outputs. GPU-to-CPU fallback is not silently
  enabled during an explicitly requested MPS run.

This is a small conditional convolutional GAN, not an exact reproduction of a
published DCGAN architecture. The conditioning follows [Mirza and Osindero's
conditional GAN formulation](https://arxiv.org/abs/1411.1784); alternating updates
and optimizer choices are informed by the [official PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

## Checkpoint selection and generation

Adversarial loss is not a reliable quality ranking. The GAN instead selects the
lowest fixed validation pooled-feature Wasserstein score. At most 512 training
and 512 validation images are selected once, stratified when conditional. The
feature scaler fits only the training subset. Fixed latent draws and validation
labels are reused each epoch. The score averages aggregate discrepancy with
per-class discrepancies when conditional. `patience` applies to this score.
The selected indices, latents, labels and fitted scale are saved in
`checkpoint_selection.npz`. Indices are positions within the recorded execution
subsets in `splits.npz`, not necessarily original dataset indices.

This is an inexpensive heuristic on pooled image statistics, not FID, a
likelihood, a perceptual score or biological validation. Reusing validation for
checkpoint selection makes it development evidence; final tests remain reserved.
GAN and VAE training/checkpoint scores are not directly comparable. Their final
quality and augmentation evaluations use the same existing evaluator contract.

`best.pt` and `last.pt` save both network states and both optimizer states, plus
configuration, classes, preprocessing and plan provenance. Samples use new
seeded normal latent draws; tanh outputs are scaled, rounded and clipped to
uint8. Conditional samples cycle evenly over observed training classes. No
Gaussian observation noise is added, and there is no `decoder_means` artifact.
The model has no reconstruction objective or ELBO; reports never label GAN
losses as either.

## Evaluation and artifacts

GAN outputs pass through the same structural, pooled-feature fidelity, diversity,
training-reference overlap and per-class checks as image VAE outputs. For
augmentation, the same real-only, real-plus-synthetic and class-matched real
resampling logistic probes are used. There are no relaxed GAN acceptance rules.
The evaluator accepts `image_gan` alongside existing models and preserves the
final-test guard. Generated grid, learning curves, full histories, checkpoints,
source hashes, split records and AI interpretations remain reviewable.

## Verification

All 51 tests passed. New tests verify detached fake gradients, actual updates to
both networks, discriminator freezing during generator updates, conditional and
unconditional output shapes, correct pixel scaling, VAE/GAN joint planning,
rejection of GAN representation learning, checkpoint reload, exact reproduction
of saved CPU samples from the best checkpoint, label balance, and real downstream
evaluation. Object-valued test arrays in the integration fixture cannot be read
by the loader; successful training/evaluation confirms test isolation.

A live AI plan selected a conditional GAN with latent 16, hidden 64, batch 32,
learning rate 0.0002, two epochs, patience two, and 32 generated samples. The
explicit executor pilot used 512 PathMNIST training and 128 validation images.
It ran on MPS, completed both epochs, chose epoch two, and generated 32 images.
Supervised wall time was 17.4 seconds and peak process RSS 0.54 GiB (not total
unified/GPU memory). The two-epoch images still show strong grid-like artifacts;
this check establishes execution capability, not image usefulness or a performance
advantage over the VAE. A longer, budgeted comparison remains to be performed.

Follow-up: the [fuller GAN validation experiment](full_gan_experiment.md) has now
completed. It provides the longer-run findings and a descriptive comparison to
the saved VAE; the pilot findings above remain a record of implementation verification.

Live evaluation and AI interpretation completed. Structural validity passed,
but fidelity and diversity screens failed. Validation macro-F1 was 0.595 for
real-only versus 0.553 with synthetic augmentation; the paired difference interval
[-0.0993, 0.0056] includes zero. This pilot establishes no augmentation benefit
and receives `insufficient_evidence`. No final-test data were used. All 79
recorded Project 1 source/document hashes remain unchanged; the machine-readable
record is `outputs/gan_milestone_verification.json`.

Review the [AI plan](../outputs/pathmnist_gan_plan_v1/modeling_plan.md),
[training record](../outputs/pathmnist_gan_training_pilot_v1/execution_report.md),
[generated grid](../outputs/pathmnist_gan_training_pilot_v1/image_gan/generated_grid.png),
and [evaluation report](../outputs/pathmnist_gan_evaluation_pilot_v1/evaluation_report.md).
