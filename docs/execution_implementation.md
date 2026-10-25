# Connecting modeling plans to execution tools

The execution command consumes a saved `modeling_plan.json` and its adjacent
`input_profile.json`. It performs no new AI planning and does not modify the plan.
All implementation and verification are confined to Project 2.

## Code map

| File | Responsibility |
|---|---|
| `genmodel_agent/capabilities.py` | Versioned trainer contracts and availability |
| `genmodel_agent/execution.py` | Plan revalidation, effective pilot plan, per-model process supervision, aggregate report |
| `genmodel_agent/training_data.py` | Source verification, splits, training-only gene selection, minibatch conversion |
| `genmodel_agent/models.py` | Convolutional image VAE, count VAE, NB2 log probability |
| `genmodel_agent/trainer.py` | Device probe, ELBO optimization, validation, checkpoints, generation, plots and embeddings |
| `tests/test_execution.py` | Likelihood checks, leakage/source protection, training, checkpoints and timeout handling |

## Available tools

Image/count VAEs, the image GAN, and three standalone expression flows are implemented. The legacy generic continuous_flow name remains unavailable. See gan_implementation.md for GAN-specific training and checkpoint rules. No silent model
substitution occurs. Multiple model branches are processed sequentially; failures
are retained while remaining branches can continue. For now the implemented catalog
contains two VAE variants, not multiple executable generative families.

Structured fields control hyperparameters. The versioned contracts define the
optimizer, likelihood details, network layout, gene-ranking algorithm and sampling
policy. Free-form rationale is explanatory, not executable code. Older plans can
be executed after revalidation using these contracts; their archived text is retained.
No arbitrary Python emitted by the LLM is evaluated.

## Image VAE

The encoder uses two stride-2 convolutions followed by a hidden projection and
Gaussian latent mean/log-variance heads. The decoder uses a hidden projection and
two transposed convolutions. `hidden_dim` controls the projection width and convolution
channels; `latent_dim` controls the bottleneck. One-hot labels are concatenated at
encoder/decoder hidden inputs when `conditional=true`.

Input support is deliberately bounded: uint8 MedMNIST images, one or three channels,
height/width divisible by four, up to 64. Pixel scaling is fixed at /255, so it needs
no estimated statistics. Gaussian observation standard deviation is fixed at 0.1.
The loss is the per-observation summed Gaussian negative log likelihood plus KL to
a standard-normal prior (beta=1). Continuous densities can exceed one, so negative
Gaussian NLL/ELBO values are valid and not themselves a bug or a quality score.

Samples draw a new latent from the prior and observation noise from the Gaussian
decoder, then clip to [0,1] and quantize to uint8. `decoder_means` are saved separately
as float32 images. The clipping is an output convention; the fitted likelihood is
an untruncated Gaussian. Conditional generation cycles evenly through observed
classes; this is an augmentation policy, not exact minority equalization.

## Count VAE

Cells are split before any fitted preprocessing. Nonconstant genes are ranked using
training-only log1p variance/mean dispersion, standardized within equal-frequency
mean bins (at most 20 bins). The requested number of genes is selected exactly;
insufficient eligible genes produce a failure rather than silently changing the plan.
This is a transparent heuristic, not an exact implementation of Scanpy's HVG routine.

Raw selected-gene counts are the decoder targets. Only the encoder applies log1p.
The decoder predicts gene proportions using softmax, multiplied by the observed
selected-panel count total. Gene-wise dispersion is learned with a positive softplus
parameterization. The NB2 model has variance mu + mu²/theta, with KL to a standard
normal latent prior. It does not treat observed zeros as missing values.

Generation draws prior latents, resamples selected-panel library sizes from training
cells, and samples a negative binomial using a gamma-Poisson mixture on CPU. Synthetic
counts are int64 and nonnegative. Expected counts, gene IDs and conditioning library
sizes are also saved. Sampled totals fluctuate around the conditioning total.
Outputs cover only the selected gene panel, not the full transcriptome. The encoder
and decoder do not model batch effects or guarantee biological identity preservation.

## Splits and source integrity

The input hashes and dimensions must match the saved profile. A relocated dataset
can be supplied with `--dataset` if hashes match. PathMNIST uses publisher training
and validation arrays; only the existence of test keys is checked. Test arrays are
not loaded. PBMC stores disjoint train/validation/test indices and uses only training
cells to fit gene selection. Counts are loaded as a sparse matrix; conversion to dense
happens per minibatch after feature selection. Test evaluation is a later stage.

Conditional label encodings are fitted from training labels. Unseen validation
classes fail. The existing raw PBMC 10x input has no labels, so its current plan is
unconditional. H5AD and generic tables remain inspection-only for this milestone.

## Training, resources and reproducibility

- Adam uses the plan's learning rate, defaults betas=(0.9,0.999), float32 and gradient clipping at norm 10.
- Validation uses fixed-seed Gaussian noise for a comparable single-sample ELBO estimate each epoch.
- Early stopping follows plan patience; the lowest complete validation loss selects `best.pt`.
- `last.pt` captures the last complete epoch. Checkpoints include optimizer state, but exact interrupted-run resume is not implemented.
- `auto` probes the actual model's forward and backward pass on MPS/CUDA when available, then CPU. All probe failures are recorded. Mid-run numerical/backend errors fail that branch, rather than silently changing the model or device.
- Each branch reserves 20% of its planned time allowance for artifacts; preprocessing also consumes the allowance. It may stop before the planned epoch count. No completed validation epoch means no usable checkpoint.
- The worker checks wall time, process RSS and MPS driver memory at batch boundaries. A parent polls RSS/time every 0.2 seconds, terminating a stuck worker after the allowance plus 15 seconds of shutdown/import overhead. These are monitored limits, not exact hard OS allocations; transient peaks and combined CPU/GPU allocation can differ from RSS.
- Seeds, source hashes, split indices, package versions, code hashes and explicit pilot overrides are saved. Cross-device bitwise identity is not promised. Saved models can be loaded with `torch.load(..., weights_only=True, map_location='cpu')`.

## Verification results

22 tests passed, including comparison of the negative-binomial log likelihood to
SciPy, finite gradients and parameter updates, checkpoint loading, source-mutation
rejection, unavailable-trainer rejection, and parent timeout termination. A test with
unloadable object-valued test arrays confirms image training does not access test data.
A separate test changes only held-out count values and confirms fitted gene selection
and training values stay unchanged.

Real-data MPS pilots (two epochs; 512 training / 128 validation observations):

| Model | Saved outputs | Observed worker wall time | Peak worker RSS |
|---|---|---:|---:|
| Conditional image VAE | 32 images, 28×28×3 | 19.4 s | 0.58 GiB |
| Count VAE | 32 profiles × 1,000 genes | 18.0 s | 0.64 GiB |

These timings are single pilot observations, not comparative benchmarks or full-run
estimates. CPU pilots also completed. MPS is available outside the desktop command
sandbox; restricted processes reported it unavailable. This difference is recorded
in device-selection outputs. The two-epoch image preview is visibly undertrained;
it must not be interpreted as evidence of useful histology augmentation.

## What remains

Quantitative quality diagnostics, downstream baseline comparisons and result-dependent recommendations are now implemented; see [the evaluation guide](evaluation_implementation.md). Full experiments and final assignment reports remain pending.
The image GAN and standalone expression flows add executable model families. Full experiments
should follow evaluation implementation; two-epoch pilot samples are not final results.

Implementation references: [PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html),
[PyTorch probability distributions](https://docs.pytorch.org/docs/2.14/distributions.html).

See [normalizing-flow implementation](normalizing_flows_implementation.md) for flow training and continuous-output evaluation. Earlier VAE verification records above remain historical.
