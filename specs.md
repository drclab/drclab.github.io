# Repository Specifications: Dr. Codex Laborer (drclab.github.io)

## 1. Project Overview
This repository hosts a Hugo-based static site titled **"Dr. Codex Laborer"** (Bayesian Data Codexer). The site acts as a research hub, blog, and computational notebook repository heavily featuring data science, statistics, R, and Python.

## 2. Core Technologies & Stack
- **Static Site Generator:** Hugo
- **Theme:** [Gokarna](https://github.com/526avijitgupta/gokarna) (using Monokai for syntax highlighting)
- **Content Format:** Markdown with TOML front matter
- **Math Rendering:** Supported via Goldmark passthrough extensions (allows LaTeX `$`, `$$`, `\(`, `\[` notation)
- **Languages / Environments:**
  - **R:** Used for data simulation and statistical modeling (custom R data histories and `.RData` present, plus scripts).
  - **Python:** Virtual environments (`venv`) and custom image extraction scripts.
  - **Shell:** Custom automation scripts.

## 3. Directory Structure
The repository follows standard Hugo conventions with additional specialized directories:

- `content/`: Holds all authored Markdown pages. Paths mirror URL paths.
  - `posts/`: Primary blog posts/research notes (~205+ items).
  - `ipynb/`: Jupyter notebook exports/references (~70+ items).
  - `pdf/`, `docker/`, `scripts/`, `references/`: Organized topic or asset-specific content.
- `scripts/`: Custom data processing and build automation scripts (e.g., Python image extractors, R data simulators).
- `static/`: Unprocessed static assets (e.g., images, downloads). Rooted at `/` during generation.
- `layouts/`: Custom template overrides (HTML) specifically for sections or distinct layouts.
- `archetypes/`: Default templates and front matter configurations for scaffolding new content.
- `themes/`: Houses the Gokarna Hugo theme and potentially others.

## 4. Configuration Details (`config.toml`)
- Base language is `en-us`.
- Enables Raw HTML natively (`unsafe = true` renderer).
- Primary navigational paths map to the author's external site (`www.dulun.com`), main Posts (`/posts/`), and Tags (`/tags/`).
- Markdown configuration supports complex mathematical typesetting directly.

## 5. Coding Style & Content Guidelines
Based on the repository's rules (`AGENTS.md`):
- Write declarative, scope-prefixed commit subjects (e.g., `content: add spring workshop recap`).
- Use **TOML** for all front matter and maintain lowercase, hyphenated keys (e.g., `show-related = true`).
- Use lowercase, hyphenated slugs for content filenames (e.g., `about-team.md`).
- Ensure static asset filenames are web-safe (lowercase, no spaces).

## 6. Build, Test & Development Workflow
**Local Development:**
```bash
hugo server --buildDrafts --buildFuture
```
**(Tip: Use `draft: true` in the front matter to keep posts out of production.)**

**Content Scaffolding:**
```bash
hugo new content/posts/my-new-post.md
```

**Production Build:**
```bash
# Clear old builds to prevent staleness
rm -rf public/
hugo --minify
```
_Note: Address any build warnings immediately, as they are treated as blockers for commits._
