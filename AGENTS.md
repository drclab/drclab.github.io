# Repository Guidelines

This repository hosts a Hugo-based site. Use this guide to keep contributions fast, predictable, and easy to review.

## Project Structure & Module Organization
- `content/` holds Markdown pages; mirror URL paths (e.g., `content/blog/post.md`).
- `layouts/` overrides theme templates; add section-specific templates under matching subfolders.
- `archetypes/` defines front matter defaults; update `archetypes/default.md` when adding new taxonomies.
- `static/` serves unprocessed assets (images, downloads); reference files with `/`-rooted URLs.
- `data/` stores TOML/JSON/YAML used by templates; keep structured content out of `config.toml`.

## Build, Test, and Development Commands
- `hugo server --buildDrafts --buildFuture` launches the live preview with drafts and upcoming posts.
- `hugo --minify` produces a production build in `public/`; run before publishing.
- `hugo new content/blog/my-post.md` scaffolds a Markdown page with the default archetype.

## Coding Style & Naming Conventions
- Prefer Markdown with front matter in TOML; keep keys lowercase with hyphenated names (`show-related = true`).
- Follow 2-space indentation inside templates and front matter for readability.
- Name content files with lowercase, hyphen-separated slugs (`content/pages/about-team.md`).
- Keep static asset filenames web-safe (lowercase, no spaces) and group by feature (`static/img/events/`).

## Testing Guidelines
- Build with `hugo --minify` and resolve any warnings; treat warnings as blockers.
- Spot-check generated pages in `public/` for broken links or missing assets before submitting.
- If you add scripts or embeds, test them in `hugo server` to confirm they run without console errors.

## Commit & Pull Request Guidelines
- Write imperative, scope-prefixed commit subjects (e.g., `content: add spring workshop recap`).
- Limit commits to logical units; include short context in the body when configuration changes or migrations occur.
- PRs should link relevant issues, describe visible changes, and include screenshots or GIFs for UI-affecting work.
- Note any follow-up tasks or TODOs in the PR description so they can be triaged.

## Configuration & Deployment Tips
- Update `config.toml` for global settings; document any new params in the PR to keep reviewers aligned.
- Before deploying, clear prior builds (`rm -rf public/`) to avoid stale files, then rerun `hugo --minify`.
- Use draft mode (`draft: true` in front matter) for in-progress pages; remove the flag when ready to publish.
