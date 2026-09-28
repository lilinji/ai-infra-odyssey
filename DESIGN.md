# Design System Specification (Anthropic Claude Official)

This document defines the visual system reverse-engineered from Anthropic's official Claude Platform Docs (`https://platform.claude.com/docs/`).

## 1. Visual Philosophy

- **Quiet & Calm**: Minimal ornamentation, ample whitespace, high legibility.
- **Warm Editorial**: Soft warm ivory/cream background in light mode, deep warm charcoal (never dead black `#000000`) in dark mode, restrained terracotta accents.
- **Hierarchical Separation**: Subtle contrast between the sidebar navigation and main reading canvas.
- **Technical Precision**: Diagrams explain system architecture and flow without decorative visual clutter.

## 2. Official Color Palette & Tokens

### Light Mode
| Token | Hex Value | Usage |
| :--- | :--- | :--- |
| `primary` | `#DA7756` | Official Claude terracotta accent, active links, primary CTA |
| `primary-hover` | `#C26343` | Hover state for primary interactive elements |
| `bg-page` | `#FAF9F5` | Warm cream page background |
| `bg-sidebar` | `#F3F0E8` | Warm stone sidebar background (subtle depth separation) |
| `sidebar-active` | `#E5E0D8` | Active sidebar item surface |
| `surface` | `#FFFFFF` | Card surface, modal, popover |
| `text-primary` | `#181816` | Main body copy and headings (warm near-black) |
| `text-muted` | `#6E6B65` | Secondary descriptions, captions, and metadata |
| `border` | `#E5E2D9` | Container borders and divider lines |
| `code-inline-bg` | `#F0ECE1` | Inline `code` background |
| `codeblock-bg` | `#1E1E1C` | High-contrast code block background |

### Dark Mode
| Token | Hex Value | Usage |
| :--- | :--- | :--- |
| `primary` | `#DA7756` | Official Claude terracotta accent (vibrant against dark) |
| `bg-page` | `#181816` | Deep warm charcoal body background (never pure `#000000`) |
| `bg-sidebar` | `#141413` | Deepest charcoal sidebar background |
| `sidebar-active` | `#2C2B28` | Active sidebar item surface |
| `surface` | `#22211F` | Elevated card surfaces and panels |
| `text-primary` | `#F3F2EE` | High-contrast warm off-white text |
| `text-muted` | `#A3A097` | Muted slate secondary copy |
| `border` | `#33322E` | Subtle dark container borders |
| `code-inline-bg` | `#262523` | Inline `code` background |
| `codeblock-bg` | `#111110` | Embedded terminal/code canvas |
| `codeblock-border` | `#2E2D2A` | Code block frame border |

## 3. Typography

- **Body & Headings**: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif
- **Code & Identifiers**: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace
- **Heading Casing**: Always use sentence case for headings (e.g. `## Configure authentication`).

## 4. Components

### Primary Buttons
- Background: `#DA7756`
- Text: `#FFFFFF`
- Radius: `6px`
- Padding: `8px 16px`
- Border: none

### Cards
- Light: `#FFFFFF` background with `1px solid #E5E2D9` border.
- Dark: `#22211F` background with `1px solid #33322E` border.
- Radius: `8px`

### Callouts
- Light: subtle warm tint with `#DA7756` left border.
- Dark: `#22211F` surface with `#DA7756` accent line.
