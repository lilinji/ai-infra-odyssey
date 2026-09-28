---
name: mintlify-claude-docs
description: Build, write, review, migrate, and maintain Mintlify documentation in a restrained Claude Docs-inspired style. Use when creating or editing docs, MDX pages, docs.json, style-guide.md, DESIGN.md, documentation navigation, API references, tutorials, how-to guides, explanations, changelogs, or AI-readable documentation.
---

# Mintlify + Claude-style documentation

## 1. Purpose

This skill defines how to build a complete documentation system using:

- Mintlify as the published documentation platform
- `docs/docs.json` as the Mintlify site configuration
- `docs/style-guide.md` as the writing and editorial source of truth
- `DESIGN.md` as the visual language source of truth
- Diátaxis as the documentation content model
- `CLAUDE.md` as the repository-level instruction file
- This skill as the execution workflow for documentation agents

The goal is not to copy any company’s private brand system.

The goal is to create documentation with the following qualities:

- Calm
- Precise
- Technical
- Restrained
- Human
- Easy to scan
- Easy to maintain
- Easy for both humans and AI systems to understand

## 2. Responsibility boundaries

Never merge all documentation concerns into one file.

### `docs/docs.json`

`docs/docs.json` controls the published Mintlify site.

Use it for:

- Site name
- Theme
- Color configuration
- Logo
- Favicon
- Navigation
- Top navigation
- Footer links
- Search configuration
- API playground configuration
- Analytics configuration
- Site-level behavior
- Other supported Mintlify configuration

Do not put the following in `docs/docs.json`:

- Writing rules
- Voice and tone
- Page templates
- Editorial advice
- Design philosophy
- Agent instructions
- Product explanations

### `docs/style-guide.md`

`docs/style-guide.md` controls how documentation is written.

Use it for:

- Audience
- Voice
- Tone
- Sentence structure
- Heading style
- Page structure
- Content types
- Code example rules
- Link rules
- Callout rules
- Terminology
- Prohibited patterns
- Review criteria

Do not put the following in `docs/style-guide.md`:

- Mintlify theme configuration
- CSS variables
- Logo paths
- UI design tokens
- Marketing page layout rules
- Illustration style
- Navigation JSON

### `DESIGN.md`

`DESIGN.md` controls the visual system used by AI when generating:

- Product UI
- Landing pages
- Marketing pages
- Illustrations
- Diagrams
- Covers
- Screenshots
- Visual examples
- Interface mockups

Use it for:

- Colors
- Typography
- Spacing
- Radius
- Borders
- Shadows
- Components
- Layout principles
- Visual do/don’t rules
- Diagram rules
- Image direction

Do not put the following in `DESIGN.md`:

- How to write a tutorial
- How to title a how-to guide
- API reference structure
- Navigation taxonomy
- Product terminology definitions
- Prose style rules

### `CLAUDE.md`

`CLAUDE.md` should contain repository-level instructions and pointers.

It should not duplicate the entire content of this skill.

It should tell the agent:

- Which files are authoritative
- What to read first
- Which page types are allowed
- Which commands to run
- Which files must not be mixed

### This skill

This skill defines how an AI agent should:

- Inspect a documentation repository
- Decide what to change
- Create or edit documentation
- Choose a page type
- Apply the writing style
- Update Mintlify navigation
- Review quality
- Preserve existing behavior
- Avoid inventing unsupported features

## 3. Required reading order

Before creating or editing a documentation page, read:

1. `CLAUDE.md`, if present
2. `docs/docs.json`
3. `docs/style-guide.md`
4. `docs/content-types.md`, if present
5. The relevant existing documentation pages
6. The related source code, API schema, or product specification when accuracy depends on it

Before generating a UI, visual asset, landing page, cover, or diagram, read:

1. `CLAUDE.md`
2. `DESIGN.md`
3. Any existing visual examples or design assets
4. The relevant product requirements

Before changing navigation or site configuration, read:

1. `docs/docs.json`
2. Existing navigation structure
3. Mintlify configuration schema or project-specific conventions
4. `CLAUDE.md`

Do not assume that a previous implementation is correct. Inspect the repository first.

## 4. Operating principles

### 4.1 Preserve the user's intent

When editing a page:

- Preserve the original task
- Preserve verified technical facts
- Preserve valid examples
- Improve clarity without changing meaning
- Avoid unrelated rewrites
- Avoid changing URLs unless necessary
- Avoid renaming pages without checking inbound links

### 4.2 Do not invent product behavior

Never invent:

- API endpoints
- Parameters
- Response fields
- CLI commands
- Configuration options
- Product limits
- Authentication methods
- UI labels
- Feature availability
- Release dates
- Error behavior
- SDK methods
- Permissions
- Pricing
- Regional availability

When information is missing:

- Inspect the source
- Search the repository
- Check schemas or type definitions
- Mark the uncertainty
- Ask for clarification if the missing fact affects correctness

Do not convert an assumption into documentation.

### 4.3 Prefer the smallest useful page

A good documentation page answers one primary question.

Do not add content merely to make the page longer.

Use additional pages when:

- The page has more than one user goal
- Tutorial and reference material are mixed
- Conceptual explanation interrupts a procedure
- One section is useful to a different audience
- The API reference has grown beyond scanability

### 4.4 Use progressive disclosure

Show the minimum information needed first.

Then link to:

- Deeper explanations
- Complete references
- Advanced configuration
- Troubleshooting
- Related concepts
- Migration notes

Do not put every possible detail in the first paragraph.

## 5. Documentation content model

Every page must have one primary type.

Allowed types:

1. Tutorial
2. How-to guide
3. Reference
4. Explanation

These correspond to the Diátaxis model.

Do not invent additional top-level page types unless the repository explicitly defines them.

## 6. Tutorial pages

### Purpose

A tutorial teaches a beginner through a complete, successful path.

A tutorial should:

- Start from a known state
- Use a clear sequence
- Minimize branching
- Produce a visible result
- Avoid covering every possible option
- End with a useful next step

### Tutorial title patterns

Good:

- Create your first project
- Build a basic integration
- Send your first request
- Deploy your first application

Avoid:

- Everything you need to know about projects
- A complete introduction to the platform
- Getting started with all features

### Tutorial structure

```mdx
---
title: Create your first project
description: Create a project and send your first test request.
---

Create a project and send your first test request.

## Before you start

List the minimum requirements.

## Create a project

Give the first action.

## Configure the project

Continue with the required setup.

## Send a request

Show a complete, copyable example.

## Verify the result

Describe what success looks like.

## Next steps

Link to the next logical guide.
```

### Tutorial rules

* Do not explain every alternative path
* Do not include a complete parameter reference
* Do not assume prior product knowledge
* Keep the path linear
* Use concrete checkpoints
* State expected results after meaningful steps

## 7. How-to guide pages

### Purpose

A how-to guide helps an existing user solve a specific problem.

A how-to guide should:

* Start with the problem
* Assume basic familiarity
* Focus on execution
* Avoid broad conceptual essays
* Include verification or troubleshooting

### How-to title patterns

Good:

* Configure authentication
* Handle failed requests
* Add a webhook endpoint
* Set a request timeout
* Export project data

Avoid:

* Authentication
* Webhooks in general
* All possible request settings

### How-to structure

```mdx
---
title: Configure authentication
description: Configure authentication for requests to the API.
---

Configure authentication for requests to the API.

## Before you start

List the prerequisites.

## Configure authentication

Provide the direct procedure.

## Verify authentication

Explain how to confirm success.

## Troubleshooting

List only relevant failure cases.

## Next steps

Link to related work.
```

### How-to rules

* Start with the action
* Use numbered steps for sequential actions
* Use bullets for non-sequential options
* Explain only the concepts needed to complete the task
* Include a working command or example when applicable
* Avoid vague steps such as “configure as needed”

## 8. Reference pages

### Purpose

A reference page provides exact, searchable information.

Use reference pages for:

* API endpoints
* SDK methods
* CLI commands
* Configuration options
* Environment variables
* Error codes
* Webhook events
* Schema definitions
* Object fields
* Limits
* Supported values

### Reference structure

```mdx
---
title: Create a project
description: Create a new project with the API.
---

Create a new project.

## Request

Show the method and path.

## Authentication

Explain required authentication.

## Parameters

Use a table or structured sections.

## Request example

Provide a complete request.

## Response

Show the response schema or example.

## Errors

Document known errors.

## Limits

Document verified limits only.
```

### Reference rules

* Prefer tables for compact, repeated facts
* Use consistent parameter names
* Document required and optional fields
* Document defaults
* Document valid values
* Document nullability
* Document response types
* Separate examples from normative definitions
* Never use prose to hide required constraints

### Parameter table

Use this format when appropriate:

```md
| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `name` | string | Yes | — | Name of the project. |
| `region` | string | No | `us` | Deployment region. |
```

If a field has no default, use `—`.

## 9. Explanation pages

### Purpose

An explanation page builds understanding.

Use explanation pages for:

* Architecture
* Mental models
* Design decisions
* Trade-offs
* Lifecycle concepts
* Security models
* Data flow
* System boundaries
* Why a feature works in a specific way

### Explanation structure

```mdx
---
title: How projects work
description: Understand how projects group resources and define access boundaries.
---

Projects group related resources and define their access boundaries.

## Mental model

Explain the central concept.

## How it works

Describe the behavior.

## Lifecycle

Describe the states or sequence.

## Trade-offs

Explain important decisions or constraints.

## Related guides

Link to procedures and references.
```

### Explanation rules

* Explain the model before the implementation details
* State trade-offs honestly
* Avoid turning the page into marketing copy
* Link to procedures instead of embedding every instruction
* Use diagrams when relationships are easier to show than describe

## 10. Voice and tone

The documentation should feel:

* Calm
* Direct
* Precise
* Helpful
* Neutral
* Confident
* Technically grounded
* Human without being casual

Use:

* Second person
* Active voice
* Present tense
* Imperative verbs for procedures
* Short paragraphs
* Concrete nouns
* Specific verbs

### Good

```text
Create an API key and store it in your environment.
```

```text
The request returns a signed token.
```

```text
Select Create project.
```

### Avoid

```text
In this section, we are going to walk you through how you can create an API key.
```

```text
You should now be able to see that the request will return a signed token.
```

```text
Simply click the button to easily create your project.
```

## 11. Prohibited writing patterns

Do not use:

* Generic welcome paragraphs
* Marketing slogans
* Hype
* Excessive exclamation marks
* “Seamless”
* “Powerful”
* “Effortless”
* “Best-in-class”
* “Revolutionary”
* “Simply”
* “Just”
* “Obviously”
* “As you can see”
* “In this section”
* “In order to”
* “We are going to”
* “Let’s dive in”
* “Now you should be able to”
* “Click here”
* Unsupported guarantees
* Unverified limits
* Unreleased features
* Fake screenshots
* Placeholder content presented as final documentation

## 12. Headings

Use sentence case.

Good:

```md
## Configure authentication
```

Avoid:

```md
## Configure Authentication
```

Good:

```md
## Handle failed requests
```

Avoid:

```md
## HANDLE FAILED REQUESTS
```

### Heading rules

* Use `#` for the page title
* Use `##` for major sections
* Use `###` for subsections
* Do not skip heading levels without a reason
* Do not create headings that contain only generic words such as “Overview” when a specific heading is possible
* Keep headings concise
* Use verbs for procedures
* Use nouns for reference sections

## 13. Frontmatter

Every published page must include:

```yaml
---
title: Clear sentence-case title
description: One sentence explaining the reader outcome.
---
```

### Title rules

A title must be:

* Specific
* Searchable
* Short
* Sentence case
* Consistent with the page type

### Description rules

A description must explain:

* What the reader can do
* What the reader can understand
* What information the page provides

Good:

```yaml
description: Create a project and send your first test request.
```

```yaml
description: Configure retry behavior for failed API requests.
```

Bad:

```yaml
description: Documentation for projects.
```

```yaml
description: This page explains authentication.
```

## 14. Page opening

The first paragraph should state the outcome.

Good:

```md
Configure authentication for requests to the API.
```

```md
Projects group related resources and define their access boundaries.
```

Avoid:

```md
Welcome to the projects documentation.
```

```md
In this guide, we will learn more about projects.
```

Do not repeat the title word-for-word unless it is the clearest opening.

## 15. Instructions and steps

Use numbered lists when order matters.

```md
1. Open the project settings.
2. Select **API keys**.
3. Select **Create key**.
4. Copy the key and store it securely.
```

Use bullet lists when order does not matter.

```md
- An API key
- A project
- A supported runtime
```

Each numbered step should contain one primary action.

Avoid:

```md
1. Open the dashboard, select your project, configure the API key, and restart the application.
```

Prefer:

```md
1. Open the dashboard.
2. Select your project.
3. Configure the API key.
4. Restart the application.
```

## 16. UI labels

Use bold text for exact UI labels.

```md
Select **Create project**.
```

```md
Open **Settings** and select **API keys**.
```

Do not bold general concepts.

```md
The project uses an API key.
```

If the product UI uses a specific capitalization, preserve it.

Do not invent UI labels.

## 17. Code examples

Code examples must be:

* Correct
* Runnable when possible
* Minimal
* Complete enough to understand
* Consistent with the current API
* Safe to copy
* Clearly labeled by language

### Code rules

* Use fenced code blocks
* Include required imports when relevant
* Include required headers
* Include realistic placeholders
* Use environment variables for secrets
* Never include real secrets
* Do not use fake fields that look official
* Do not show incomplete syntax as a complete example

### Good

```bash
curl https://api.example.com/v1/projects \
  -H "Authorization: Bearer $API_KEY"
```

### Avoid

```bash
curl ...
```

unless the omitted content is intentional and explained.

### Secrets

Use:

```bash
export API_KEY="your-api-key"
```

Do not use:

```bash
export API_KEY="sk_live_actual_secret"
```

## 18. API documentation

API documentation must identify:

* HTTP method
* Endpoint
* Authentication
* Headers
* Path parameters
* Query parameters
* Request body
* Response body
* Status codes
* Errors
* Pagination
* Rate limits, if verified
* Idempotency behavior, if applicable
* Versioning behavior, if applicable

### API example

```mdx
## Request

```http
POST /v1/projects
```

## Headers

| Header          | Required | Description                 |
| --------------- | -------- | --------------------------- |
| `Authorization` | Yes      | Bearer API key.             |
| `Content-Type`  | Yes      | Must be `application/json`. |

## Body parameters

| Parameter | Type   | Required | Description          |
| --------- | ------ | -------- | -------------------- |
| `name`    | string | Yes      | Name of the project. |

## Example

```bash
curl -X POST https://api.example.com/v1/projects \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"Example project"}'
```

## Response

```json
{
  "id": "proj_123",
  "name": "Example project"
}
```
```

Do not document API behavior from memory.

Use source schemas, route definitions, generated clients, tests, or verified product behavior.

## 19. Callouts

Use callouts sparingly.

Use `<Note>` for useful context:

```mdx
<Note>
Store API keys outside your source code.
</Note>
```

Use `<Tip>` for an optional improvement:

```mdx
<Tip>
Use an environment variable so you can change the key without editing the command.
</Tip>
```

Use `<Warning>` for a real risk:

```mdx
<Warning>
Deleting a project cannot be undone.
</Warning>
```

Use `<Info>` for supplementary information:

```mdx
<Info>
The request is processed asynchronously.
</Info>
```

Do not use callouts to hide:

* Missing prerequisites
* Ambiguous instructions
* Unsupported claims
* Important core steps
* Long paragraphs

## 20. Tables

Use tables when users need to compare repeated fields.

Good uses:

* Parameters
* Configuration options
* Error codes
* Supported values
* SDK compatibility
* Required headers
* Version support

Avoid tables for:

* Long paragraphs
* Complex nested data
* Procedures
* Content that requires extensive explanation

Keep table cells concise.

## 21. Links

Use descriptive link text.

Good:

```md
Continue with [authentication](../guides/authentication).
```

Avoid:

```md
Click [here](../guides/authentication).
```

Link to:

* The next logical task
* The relevant reference
* The prerequisite
* The conceptual explanation
* The troubleshooting page

Do not link every noun.

Before finishing, check:

* Relative paths
* Anchor names
* Case sensitivity
* Moved pages
* Removed pages
* External URLs

## 22. Navigation

Navigation should follow the user's mental model, not the company's internal organization.

Recommended top-level groups:

1. Start here
2. Tutorials
3. Guides
4. Concepts
5. Reference
6. Changelog

A product may use different names, but the purpose should remain clear.

### Good navigation

```text
Start here
Tutorials
Guides
Concepts
Reference
Changelog
```

### Avoid

```text
Platform Team
Core Services
Internal APIs
Backend Modules
Miscellaneous
```

unless those names are genuinely how users understand the product.

### Navigation rules

* Put the first successful experience near the top
* Put common tasks before advanced concepts
* Keep reference pages searchable and predictable
* Keep changelog separate from conceptual documentation
* Do not duplicate one page in several groups unless the platform requires it
* Do not add a page to navigation before confirming its path

## 23. Mintlify configuration

Use `docs/docs.json` as the site configuration source.

The configuration should generally include:

* Schema declaration when supported
* Site name
* Theme
* Colors
* Logo
* Favicon
* Navigation
* Relevant site links

A restrained Claude-inspired starting point is:

```json
{
  "$schema": "https://mintlify.com/docs.json",
  "name": "Your Documentation",
  "theme": "maple",
  "colors": {
    "primary": "#D97757",
    "light": "#FAF9F5",
    "dark": "#141413"
  }
}
```

The exact schema depends on the installed Mintlify version.

Before changing configuration:

1. Inspect the existing `docs.json`.
2. Confirm the installed or documented schema.
3. Preserve supported existing settings.
4. Change only the fields required for the task.
5. Validate JSON syntax.
6. Check the site build if available.

Do not put `DESIGN.md` instructions into `docs.json`.

Do not assume that Mintlify reads `DESIGN.md`.

Mintlify uses its own configuration. The design file is for agents generating visual work.

## 24. Claude-inspired visual direction

The visual language should be:

* Warm
* Quiet
* Minimal
* Editorial
* Technical
* Spacious
* Legible

Use:

* Warm off-white backgrounds
* Near-black text
* Terracotta as a focused accent
* Thin borders
* Limited shadows
* Simple diagrams
* Clear hierarchy
* Restrained component decoration

Avoid:

* Neon colors
* Heavy gradients
* Glassmorphism
* Excessive cards
* Huge decorative headings
* Rainbow color systems
* Dense shadows
* Overly rounded interfaces
* Visual noise

The accent color is a guide, not a requirement to color everything.

## 25. Design token alignment

When `DESIGN.md` exists, keep its primary accent aligned with `docs/docs.json`.

Recommended baseline:

```text
Primary:           #D97757
Light background:  #FAF9F5
Dark background:   #141413
Text:              #242321
Muted text:        #706D68
Border:            #E7E3DC
Surface:           #FFFFFF
```

The exact values may be customized, but the site and generated visual assets must use the same system.

Do not create a separate palette for:

* Documentation
* Landing pages
* Product UI
* Diagrams
* Illustrations

unless the project explicitly defines separate brands.

## 26. Typography direction

Use a clean, neutral sans-serif.

Preferred characteristics:

* High readability
* Moderate x-height
* Clear hierarchy
* No decorative display font
* Monospace for code and identifiers

Suggested stack:

```css
font-family:
  Inter,
  ui-sans-serif,
  system-ui,
  -apple-system,
  BlinkMacSystemFont,
  "Segoe UI",
  sans-serif;
```

Suggested code stack:

```css
font-family:
  "SFMono-Regular",
  Consolas,
  "Liberation Mono",
  monospace;
```

Avoid:

* Decorative fonts
* Multiple unrelated font families
* Extremely thin text
* Excessive letter spacing
* All-caps body copy

## 27. Visual components

### Buttons

Primary buttons:

* Use the terracotta accent
* Use white text
* Use a moderate radius
* Have one clear action
* Do not compete with several other primary buttons

Secondary buttons:

* Use a neutral surface
* Use a border
* Use dark text

### Cards

Use cards only to group meaningful content.

Good cards have:

* One purpose
* Clear title
* Limited text
* Consistent padding
* Quiet borders
* One relevant action

Do not wrap every paragraph in a card.

### Callouts

Use a subtle tinted surface and a narrow accent.

Avoid loud full-color panels.

### Diagrams

Diagrams should explain:

* Relationships
* Boundaries
* Flows
* Lifecycles
* Dependencies
* Data movement

Use:

* Simple boxes
* Thin lines
* One primary accent
* Clear labels
* Ample spacing

Avoid decorative complexity.

## 28. Image and illustration generation

When generating images, covers, or illustrations:

* Prefer quiet editorial compositions
* Use warm neutrals
* Use terracotta accents sparingly
* Keep one visual idea per image
* Avoid stock-photo aesthetics
* Avoid visual clutter
* Avoid unnecessary text inside images
* Avoid unreadable fake UI
* Do not invent product logos
* Do not create visual elements that imply unsupported product features

For diagrams:

* Prefer semantic shapes over decorative illustrations
* Label all important relationships
* Keep the reading direction obvious
* Use color to distinguish meaning, not decoration

## 29. Multilingual documentation

Preserve the repository’s existing language unless the task requests a translation.

When writing Chinese documentation:

* Use concise, natural Chinese
* Prefer direct verbs
* Avoid excessive literal translation
* Keep product names and code identifiers unchanged
* Use full-width punctuation in prose when consistent with the repository
* Keep code, API names, and configuration keys in their original form
* Use consistent translations for repeated terms

When writing English documentation:

* Use plain technical English
* Avoid unnecessary idioms
* Use sentence case
* Keep paragraphs short
* Prefer direct verbs

Do not mix Chinese and English arbitrarily on one page.

## 30. Versioning and compatibility

When documenting version-specific behavior:

* State the version clearly
* Use the repository's existing versioning convention
* Keep deprecated behavior separate from current behavior
* Do not describe old behavior as current
* Add migration guidance when behavior changes
* Link to release notes or changelog entries where appropriate

Use wording such as:

```md
This option is available in version 2.4 and later.
```

Do not write:

```md
This works everywhere.
```

unless that claim is verified.

## 31. Changelog pages

Changelogs should be factual and scannable.

Recommended structure:

```md
# Changelog

## 2025-01-15

### Added

- Added support for project-level API keys.

### Changed

- Updated the default request timeout.

### Fixed

- Fixed an issue with webhook retries.
```

Avoid turning changelog entries into marketing copy.

Each entry should describe:

* What changed
* Who is affected
* Whether action is required
* Where to learn more

## 32. AI-readable documentation

Documentation should be easy for AI systems to retrieve and understand.

To improve AI readability:

* Use clear page titles
* Use meaningful descriptions
* Keep one topic per page
* Use explicit headings
* Avoid vague references
* State prerequisites
* State expected results
* Use stable terminology
* Define acronyms
* Keep examples close to the explanation
* Link related pages explicitly
* Avoid hiding important facts in images
* Keep metadata accurate

Do not write artificial keyword stuffing.

Do not repeat the same phrase unnaturally for search optimization.

## 33. Migration workflow

When migrating existing docs:

### Step 1: Inventory

Identify:

* Existing pages
* Page titles
* URLs
* Navigation entries
* Duplicates
* Orphaned pages
* Broken links
* Mixed page types
* Outdated content
* Pages with missing metadata

### Step 2: Classify

Assign every page one primary type:

* Tutorial
* How-to
* Reference
* Explanation
* Changelog
* Landing/index page

Do not force changelog or index pages into the four Diátaxis types.

### Step 3: Separate mixed pages

Split pages that combine:

* Beginner setup and full reference
* Conceptual explanation and troubleshooting
* API reference and product marketing
* Installation and architecture
* Quickstart and migration guide

### Step 4: Rewrite metadata

Add:

```yaml
---
title: ...
description: ...
---
```

Descriptions should state the user outcome.

### Step 5: Rewrite the opening

Replace generic introductions with a direct purpose statement.

### Step 6: Normalize structure

Apply:

* Sentence-case headings
* Consistent section ordering
* Consistent code fences
* Consistent terminology
* Consistent callout usage

### Step 7: Update navigation

Add pages according to user tasks.

### Step 8: Validate

Check:

* Links
* JSON
* MDX syntax
* Navigation paths
* Code examples
* Product accuracy
* Duplicated content
* Missing prerequisites

## 34. New page workflow

When creating a new page:

1. Identify the user's goal.
2. Select exactly one page type.
3. Find the matching template.
4. Confirm the source of technical truth.
5. Create frontmatter.
6. Write the outcome sentence.
7. Add prerequisites.
8. Add the smallest useful content.
9. Add a verifiable result.
10. Add relevant next steps.
11. Add the page to navigation.
12. Check links.
13. Review against this skill.
14. Run available validation commands.

## 35. Existing page editing workflow

When editing an existing page:

1. Read the whole page.
2. Identify the primary page type.
3. Identify the user problem.
4. Preserve valid technical content.
5. Remove repetition.
6. Correct unsupported or stale claims.
7. Improve headings and structure.
8. Improve examples.
9. Preserve stable URLs where possible.
10. Update navigation only when necessary.
11. Review for scope creep.

Do not rewrite an entire documentation system when the task requests a small correction.

## 36. Page splitting workflow

Recommend splitting a page when:

* It serves multiple audiences
* It has more than one primary goal
* It contains unrelated troubleshooting
* It mixes setup with reference
* It has very long parameter tables and conceptual explanations
* Readers need to scan past unrelated sections
* The page has multiple independent entry points

When splitting:

* Preserve the original URL if possible
* Create clear destination pages
* Add redirects if supported
* Add links between the new pages
* Keep one canonical explanation for each concept
* Update navigation
* Check inbound links

## 37. Review checklist

Before completing any documentation task, verify:

### Structure

* [ ] The page has one clear purpose.
* [ ] The page has one primary content type.
* [ ] The page is in the correct directory.
* [ ] The page is included in navigation when appropriate.
* [ ] The page does not duplicate another page unnecessarily.

### Metadata

* [ ] `title` exists.
* [ ] `description` exists.
* [ ] The title uses sentence case.
* [ ] The description states the reader outcome.
* [ ] The metadata is accurate.

### Writing

* [ ] The opening paragraph is useful.
* [ ] The writing uses active voice.
* [ ] The writing uses direct verbs.
* [ ] Paragraphs are short.
* [ ] The tone is calm and precise.
* [ ] Marketing language has been removed.
* [ ] Unsupported claims have been removed.
* [ ] Terminology is consistent.

### Procedures

* [ ] Steps are in the correct order.
* [ ] Each step has one main action.
* [ ] Prerequisites are stated.
* [ ] Expected results are stated.
* [ ] Troubleshooting is relevant.
* [ ] UI labels match the product.

### Code

* [ ] Code fences specify a language.
* [ ] Examples are copyable.
* [ ] Secrets are not exposed.
* [ ] Required headers and imports are present.
* [ ] Examples match the documented API.
* [ ] Placeholder values are clearly placeholders.

### Links

* [ ] Relative links resolve.
* [ ] Anchors resolve.
* [ ] Link text describes the destination.
* [ ] There are no “click here” links.
* [ ] Moved or renamed pages have been checked.

### Platform

* [ ] `docs/docs.json` remains valid JSON.
* [ ] Navigation paths are valid.
* [ ] Mintlify configuration fields match the project version.
* [ ] Site-level configuration remains in `docs/docs.json`.
* [ ] Writing rules remain in `docs/style-guide.md`.
* [ ] Visual rules remain in `DESIGN.md`.

### Visual work

* [ ] Generated UI follows `DESIGN.md`.
* [ ] The primary accent is consistent.
* [ ] Visuals use restrained decoration.
* [ ] Diagrams explain rather than decorate.
* [ ] No unsupported product behavior appears in screenshots or mockups.

## 38. Response format for documentation tasks

When completing a task, report:

### Changes made

List the files changed and the purpose of each change.

### Content decision

State:

* The page type
* The intended reader
* The main user outcome

### Validation

List the checks performed:

* Frontmatter
* Links
* Navigation
* JSON
* MDX
* Code examples
* Terminology
* Product accuracy

### Open questions

List only unresolved issues that affect correctness.

Do not claim validation that was not performed.

## 39. Handling ambiguity

If the task is ambiguous:

1. Inspect the repository.
2. Infer from existing conventions.
3. Prefer the smallest safe change.
4. Preserve existing URLs and structure.
5. Ask a question only when proceeding would risk incorrect technical documentation.

Do not ask for clarification when the repository already provides enough evidence.

## 40. Final decision rules

When choices conflict, use this priority:

1. Technical correctness
2. Existing repository conventions
3. User intent
4. Clear information architecture
5. Consistent writing style
6. Visual polish

Never sacrifice technical correctness for visual similarity.

Never sacrifice clarity for brand voice.

Never place writing rules inside Mintlify configuration.

Never place site configuration inside `DESIGN.md`.

Never treat `DESIGN.md` as a replacement for `docs/style-guide.md`.

The final system should remain:

```text
Mintlify       = published site
docs.json      = site configuration
style-guide.md = writing system
DESIGN.md      = visual system
CLAUDE.md      = repository instructions
SKILL.md       = agent execution workflow
```
