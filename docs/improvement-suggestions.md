# Dulun Theme Improvement Ideas

This document collects practical enhancements you can apply to a site built with the Dulun (Hugo Coder) theme. Each idea highlights why the change matters and points to the theme files or configuration knobs you can adjust.

## 1. Refresh the Landing Experience
* **Add a hero banner with a concise value proposition.** Replace or extend `layouts/partials/home.html` so the first screen communicates who you are, what the site offers, and includes a primary call-to-action (CTA). A full-width banner, background illustration, or subtle gradient immediately differentiates your site from the vanilla theme layout.
* **Highlight featured content or projects.** Introduce a grid or carousel below the hero section that surfaces flagship posts or portfolio pieces. You can render a curated list of pages by adding a custom partial that filters on front-matter parameters (for example, `featured = true`).

## 2. Strengthen Visual Identity
* **Customize the color palette and typography.** Update the SCSS partials in `assets/scss` to reflect your brand colors and pair the default monospaced headings with a modern sans-serif or serif. Incorporate CSS variables for primary/secondary accents and apply consistent spacing scale.
* **Refine dark/light handling.** The base template already supports `light`, `dark`, and `auto` color schemes via `.Site.Params.colorScheme` in `config.toml`. Pair that with a visible toggle in the header so visitors can switch modes manually.

## 3. Improve Content Discovery
* **Implement on-site search.** Integrate a lightweight JavaScript search (e.g., Fuse.js + a generated JSON index) exposed through a header icon. You can extend the theme’s asset pipeline by adding a script to `assets/js` and rendering the search UI in `layouts/partials/header.html`.
* **Surface taxonomy hubs.** Create dedicated listing pages for tags and categories, and add teaser cards for the most popular topics to the homepage. This helps new visitors understand what you write about.

## 4. Enhance Trust and Engagement
* **Expand the About section.** Replace the brief author blurb in `layouts/partials/home/author.html` with a richer narrative, professional photo, and links to key resources (résumé, speaking, newsletter). A personable introduction increases credibility.
* **Offer multiple subscription paths.** Beyond RSS, embed a newsletter signup (Mailchimp, Buttondown, etc.) and promote it via the CTA and footer. If you enable comments (`config.toml` supports providers like Disqus, Commento, and Giscus through params), invite readers to join the conversation at the end of each post.

## 5. Optimize Performance & Media
* **Adopt responsive images.** Use Hugo’s image processing in your content templates to generate resized variants, and set `srcset`/`sizes` attributes so pages load the smallest necessary asset. Pair this with lazy loading for offscreen images.
* **Audit third-party scripts.** Consolidate analytics or tracking scripts (configured via `layouts/_default/baseof.html`) to minimize blocking requests. Prefer privacy-friendly providers already supported by the theme, such as Plausible or GoatCounter.

## 6. Bolster Accessibility & Internationalization
* **Check color contrast and keyboard flows.** Validate accessible contrast ratios after applying custom colors, ensure focus outlines are visible, and test that the navigation works with keyboard-only input.
* **Leverage multilingual support.** If you publish in multiple languages, populate `i18n/` files and configure additional languages in `config.toml` so Hugo generates localized routes and menus automatically.

These upgrades balance visual polish, content clarity, and technical robustness while staying compatible with the Dulun theme’s structure. Apply them incrementally, measuring the impact on engagement (e.g., time on page, newsletter signups) to prioritize the most valuable enhancements.
