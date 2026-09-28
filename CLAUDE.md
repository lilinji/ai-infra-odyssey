# Repository instructions

This repository uses Mintlify for published documentation.

## Documentation source of truth

- Mintlify site configuration: `docs/docs.json`
- Documentation writing rules: `docs/style-guide.md`
- Documentation content types: `docs/content-types.md`
- Visual design system: `DESIGN.md`
- Agent execution workflow: `.claude/skills/mintlify-claude-docs/SKILL.md`

## Required workflow

Before editing documentation:

1. Read this file.
2. Read `docs/docs.json`.
3. Read `docs/style-guide.md`.
4. Read `docs/content-types.md` if it exists.
5. Read the relevant existing pages.

Before generating UI, diagrams, illustrations, or landing pages:

1. Read `DESIGN.md`.
2. Reuse its visual tokens.
3. Do not invent a new visual system.

## Content types

Use exactly one of the following page types:

- Tutorial
- How-to guide
- Reference
- Explanation

Do not invent a new page type without a repository-level reason.

## Important boundaries

- Put Mintlify configuration in `docs/docs.json`.
- Put prose and editorial rules in `docs/style-guide.md`.
- Put visual rules in `DESIGN.md`.
- Do not duplicate the full contents of these files in this file.
- Do not invent undocumented APIs, features, limits, or UI behavior.
