# DSAI 601 Regression — Interactive Demos

Small, standalone D3 apps demonstrating topics from the course. Served via GitHub Pages from this repo.

## Structure

```
index.html               landing page listing all topics
topics/<topic-name>/     one self-contained demo per folder (index.html + app.js)
```

## Adding a new topic

1. `mkdir topics/<topic-name>`
2. Copy the pattern from `topics/ols-fit/` (index.html + app.js, D3 loaded via CDN)
3. Add a link + description to the root `index.html`

## Local dev

D3's data-loading functions require a real server (not `file://`):

```
npx serve .
```

## Deploy

Push to `main`. A GitHub Actions workflow (`.github/workflows/deploy-pages.yml`, at the repo root) builds and deploys the contents of this `apps/` folder to GitHub Pages — the repo itself can hold other course material (slides, assignments, etc.) alongside it without affecting the published site.
