# Repository Guidelines

## Project Structure & Module Organization
The site uses Hugo with theme code in the repository root. `content/` holds Markdown for posts and taxonomy indexes; keep front matter in TOML and place new entries under the matching topic folder. `layouts/` contains Go HTML templates and partials; reuse blocks instead of duplicating markup. Styling lives in `assets/scss/` and is compiled by Hugo Pipes alongside `assets/js/` modules; keep shared variables in `_base.scss`. Static resources such as favicons and downloads belong in `static/`. The `exampleSite/` directory mirrors a full site used for demos; update it when introducing new features.

## Build, Test, and Development Commands
Use `make demo` for a live preview with drafts (`hugo server -D`). Run `make build` to render the example site and confirm the theme builds cleanly. Before publishing, run `hugo --gc --minify --printPathWarnings` to surface unused files and warnings. Use `make release` when you need to refresh the checked-in `resources/` bundle.

## Coding Style & Naming Conventions
Adopt two-space indentation for SCSS, templates, and Markdown front matter. CSS classes follow a BEM-like scheme (`block__element--modifier`) to keep selectors predictable. Keep template logic small and favor Hugo partials for reuse; wrap template conditions in whitespace-trimmed tags (`{{- ... -}}`) when appropriate. Ensure Markdown slugs are lowercase with hyphens and store images under `static/images/<slug>/`.

## Testing Guidelines
There is no separate test harness; instead, rely on Hugo’s build. Always run `hugo --verbose` at the repository root and check the output for broken references or shortcode errors. For content changes, visit affected pages in the `make demo` server and verify syntax highlighting, dark mode, and responsive layout. Consider adding draft content under `content/posts/draft.md` until ready.

## Commit & Pull Request Guidelines
Commit subjects should be imperative and concise, mirroring existing history (“Refine home SCSS nesting…”). Explain the “why” in the body when extra context is needed. For pull requests, provide a short summary, reference related issues, list manual checks performed, and add screenshots or GIFs for visual tweaks. Request review before merging and avoid force pushes after review starts.
