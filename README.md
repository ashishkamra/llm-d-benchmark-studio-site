# llm-d Benchmark Studio — public site

**Website:** <https://ashishkamra.github.io/llm-d-benchmark-studio-site/>

This deployment-only repository hosts the generated static UI, public model
catalog/version lock, and the downloadable scripts needed to execute exported
benchmark recipes. It does not contain the private development repository's
Git history, credentials, generated datasets, or benchmark result files.

Imported results are processed in your browser, not uploaded. The demo contains
fabricated sample measurements and is labeled accordingly. Read the
[benchmark runbook and limitations](bundle/README.md) before executing a recipe.

## Hosting and updates

GitHub Pages serves the root of this repository's `main` branch. `.nojekyll`
keeps HTML, JavaScript modules, JSON catalogs, and Python downloads unchanged.
HTTPS is enforced. Pushes to this repository's `main` branch redeploy the site.

The private development repository is **not** automatically mirrored here.
To publish an update, run its tests and `scripts/build_site.py`, copy only the
reviewed generated assets into this checkout, and commit/push them here. Keep
the public GitHub and runbook links in `index.html` pointed at this repository.
Never copy the private `.git` directory, environment files, raw datasets,
benchmark results, or unrelated source files into this repository.
