"""Embed existing per-model figures without rerunning training or evaluation."""
from pathlib import Path
from urllib.parse import quote


def markdown(report_dir, attempt_dir):
    """Return portable image links for available artifacts in this attempt only."""
    root = Path(report_dir).resolve()
    attempt = Path(attempt_dir).resolve()
    models = set()
    for stage in ('execution', 'evaluation'):
        directory = attempt / stage
        if directory.is_dir():
            models.update(p.name for p in directory.iterdir() if p.is_dir())
    lines = []
    for name in sorted(models):
        training = attempt / 'execution' / name
        evaluation = attempt / 'evaluation' / name
        samples = sorted(evaluation.glob('visual_samples_*.png'),
                         key=lambda p: (len(p.stem), p.stem))
        figures = []
        if samples:
            figures.extend((p, 'Real and generated samples',
                            'Fixed-seed real/generated comparison; source indices and class labels, '
                            'when available, appear in the figure. Assigned generated labels are not '
                            'verified identities. Displaying this packet does not constitute visual review.')
                           for p in samples)
        elif (training / 'generated_grid.png').is_file():
            figures.append((training / 'generated_grid.png', 'Generated samples',
                            'Generated samples only; this is not a real/generated comparison '
                            'or evidence of downstream usefulness.'))
        figures += [
            (training / 'learning_curve.png', 'Training diagnostics',
             'Model-specific training and validation diagnostics. Better model fit does not '
             'by itself establish sample quality or downstream usefulness.'),
            (evaluation / 'quality_diagnostics.png', 'Sample-quality diagnostics',
             'Discrepancies and distances in the recorded evaluation feature space. '
             'Different metrics need not share a scale; these summaries establish neither '
             'biological validity nor a privacy guarantee.'),
        ]
        available = []
        for path, title, caption in figures:
            # Ignore missing files and symlinks outside this report's artifact tree.
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            link = quote(path.relative_to(root).as_posix(), safe='/')
            available += [f'![{title}](<{link}>)', '', caption, '']
        if available:
            lines += [f'### Visualizations: {name}', ''] + available
    return lines
