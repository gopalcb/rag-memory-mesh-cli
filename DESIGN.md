---
version: alpha
name: RAG Memory ingestion console
description: A local operator console with an indigo index masthead and a quiet working surface.
colors:
  canvas: "#edf0f7"
  paper: "#ffffff"
  navy: "#151d3e"
  primary: "#464ecd"
  muted: "#535d7c"
  line: "#dce1ee"
  signal: "#a8ebcc"
  success: "#17634b"
  error: "#a13548"
typography:
  sans:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, Liberation Mono, monospace"
rounded:
  sm: "4px"
  md: "5px"
  lg: "10px"
spacing:
  page-max: "1224px"
  section-gap: "20px"
  control-min: "44px"
components:
  button:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.paper}"
    rounded: "{rounded.md}"
    height: "{spacing.control-min}"
  page:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.navy}"
    width: "{spacing.page-max}"
  panel:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
    rounded: "{rounded.lg}"
  panel-divider:
    backgroundColor: "{colors.line}"
  connection:
    backgroundColor: "{colors.navy}"
    textColor: "{colors.signal}"
  path-field:
    typography: "{typography.mono}"
    backgroundColor: "{colors.paper}"
    textColor: "{colors.navy}"
  help-text:
    textColor: "{colors.muted}"
  feedback-success:
    textColor: "{colors.success}"
  feedback-error:
    textColor: "{colors.error}"
---

# RAG Memory ingestion console design

## Overview

The masthead's three-line index mark introduces a local instrument panel. Its operator is a developer using the English-language console to index local sources and check recent runs. The only route is an application view. Its controls and activity are the content.

Runtime CSS in [ingestion.css](ingestion_ui/static/ingestion.css) owns actual values. The front matter mirrors the principal values. Its primary color maps to CSS `--accent`.

## Colors and type

Use navy for text and the masthead, paper for data, and canvas around the working panels. Indigo identifies the primary action and selected mode. A green or red status looks different and carries a text label. Use the system sans-serif for actions and headings and the system monospace for paths and run metrics.

## Layout

The source form and watch list sit beside each other. The activity table spans the width beneath them. The page owns vertical scrolling. A large watch list or wide activity table can scroll within its panel. At narrow widths the form and watches stack, and activity retains all columns.

## Components and interaction

- The two source modes share a labeled path field, validation, and the primary action. Native radios own mode selection and native buttons own actions.
- Pausing and resuming a watch takes effect after the server confirms it. Status refresh keeps the last rendered data if the connection fails.
- Buttons and the path field have a 44 px minimum height. Panels use a 10 px radius. Dividers and fill colors establish hierarchy.
- Keep motion out of background refresh. Use visible focus and concise progress, success, and failure messages.
