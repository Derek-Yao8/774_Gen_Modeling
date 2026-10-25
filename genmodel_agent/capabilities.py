"""Versioned execution contracts. Prose never overrides these tool behaviors."""

FLOW_NAMES = ('planar_flow', 'radial_flow', 'realnvp_flow')

TRAINERS = {
    "image_vae": {
        "version": "1.0", "implemented": True,
        "architecture": "Two stride-2 convolutions, a hidden projection, diagonal-Gaussian encoder, transposed-convolution decoder.",
        "image_support": "uint8 2D images with 1 or 3 channels; height and width divisible by 4, at most 64.",
        "likelihood": "Fixed standard deviation 0.1 Gaussian on pixels scaled by 255; beta=1 ELBO.",
        "generation": "Standard-normal latent, Gaussian decoder draws clipped to [0,1]; decoder means also saved separately.",
        "conditioning": "One-hot class labels concatenated in encoder and decoder; balanced generation across observed classes.",
    },
    "count_vae": {
        "version": "1.0", "implemented": True,
        "architecture": "Two hidden layers, diagonal-Gaussian encoder, softmax proportions decoder, gene-wise learned dispersion.",
        "preprocessing": "Training-only mean-binned standardized dispersion gene selection; raw count targets; log1p counts for encoder only.",
        "likelihood": "Negative binomial with mean=selected-panel library size * decoded proportions; beta=1 ELBO.",
        "generation": "Standard-normal latent; resample selected-panel library sizes from training cells (within class if conditional); gamma-Poisson draws on CPU.",
        "conditioning": "Optional one-hot class labels in encoder and decoder; balanced generated classes.",
    },
    "image_gan": {
        "version": "1.0", "implemented": True,
        "architecture": "Latent/one-hot labels projected through two linear layers and two transposed convolutions with tanh; discriminator has two convolutions and two linear layers. No batch normalization.",
        "image_support": "Same uint8 2D 1/3-channel image layouts as image VAE; scale to [-1,1].",
        "tasks": "Generation and class balancing/augmentation only; no encoder or representation learning.",
        "likelihood": "Implicit; no likelihood, reconstruction loss or ELBO is computed.",
        "optimizer": "Separate Adam optimizers, both use plan learning_rate, betas (0.5,0.999); gradient norm clipping 10; float32. Overrides VAE defaults.",
        "training": "One discriminator update with detached fake samples followed by one non-saturating generator update with discriminator parameters frozen; stable BCE-with-logits losses.",
        "conditioning": "Optional one-hot labels in generator input and spatial label channels in discriminator. Fake training labels match each real batch; final generation cycles evenly across observed training classes.",
        "generation": "Standard-normal latent passed through generator, tanh rescaled/rounded to uint8; no extra observation noise or decoder means.",
        "validation": "Checkpoint/early stopping minimizes pooled-image feature Wasserstein distance on fixed up-to-512 training and validation subsets. Scaler fits training only; fixed latent draws match validation labels. Average aggregate discrepancy and per-class discrepancies when conditional. Same validation score each epoch, not adversarial loss or ELBO; not FID or proof of usefulness.",
    },
    "continuous_flow": {"implemented": False, "reason": "Trainer not implemented yet."},
}

COMMON = {
    "optimizer": "VAE default: Adam (betas 0.9, 0.999); gradient norm clipped to 10; float32. GAN uses its explicit per-model optimizer contract.",
    "validation": "VAE only: one fixed-seed Monte Carlo ELBO estimate per validation observation; lowest complete validation loss selects checkpoint.",
    "training": "Plan supplies latent_dim, hidden_dim, batch_size, epochs, learning_rate, patience, max_minutes, selected_features, generate_samples.",
    "resource_limits": "Cooperative batch-boundary checks plus parent wall-time/RSS supervision. MPS driver allocation is checked by the worker. These are monitored limits, not OS reservation guarantees.",
}

FLOW_EVALUATION = {
    'version':'1.0',
    'domain':'Continuous log(count + fixed per-cell Uniform[1e-6,1) jitter), on a training-selected gene panel. Not raw integer counts or normalized expression.',
    'fidelity':'Training-standardized mean marginal Wasserstein distance and correlation MAE on up to 64 training-variable genes; compare equal-size independently dequantized real references using existing 1.1x heuristic.',
    'diversity_overlap':'Same distance-ratio and training-reference overlap diagnostics in standardized continuous log-expression coordinates. No exact-zero/count-validity claims.',
    'utility':'No supervised utility for this unconditional generation objective. No biological or privacy guarantee.'}
for _name in ('planar_flow','radial_flow','realnvp_flow'):
    TRAINERS[_name] = {
        'version':'1.0','implemented':True,'family':_name,
        'architecture':'Standalone invertible data-to-standard-normal flow; learned diagonal affine transform followed by flow_layers transforms. No VAE.',
        'preprocessing':'Select 2..128 genes using training-only dispersion ranking. Deterministic per-cell uniform dequantization, log(count+U), then training-only mean/std. No library normalization. latent_dim equals selected_features.',
        'training':'Exact change-of-variables density of standardized continuous coordinates; Adam betas (0.9,0.999), gradient clip 10. Validation NLL selects best checkpoint and patience. This is not an integer count likelihood or ELBO.',
        'generation':'Normal latent, reverse transforms; planar uses 64-step monotone bisection and checks inverse residual, radial and affine coupling use analytic inverses. Undo training standardization; save continuous log-expression samples without rounding or exponentiation.',
        'tasks':'Unconditional data_generation only. Coupling hidden_dim sets MLP width; planar/radial hidden_dim=16 is an unused schema placeholder.',
        'evaluation':FLOW_EVALUATION,
        'limits':'Planar/radial limited expressiveness; low-dimensional selected panel only. Dequantization changes the target and removes exact zeros. Inverse failures/nonfinite outputs fail the run.'}
TRAINERS['continuous_flow']['reason']='Deprecated generic placeholder. Select planar_flow, radial_flow or realnvp_flow explicitly.'
