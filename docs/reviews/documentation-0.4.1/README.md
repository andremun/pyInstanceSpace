# Documentation review for 0.4.1

Maintainer review evidence for #367; these files are not included in the public
reference website. Captured on 18 October 2026.

The Python site uses the MATLAB reference's sidebar groups, header, breadcrumbs,
typography, content width and on-page navigation. Curated Python workflows and
stage references are the primary entry points. Generated API signatures use the
same layout. Python-specific settings and differences are documented explicitly.

| View | Evidence |
|---|---|
| Previously published Python landing page | [Before](published-before.jpg) |
| MATLAB v0.9.2 reference, 1280px viewport | [MATLAB reference](matlab-reference.jpg) |
| New Python landing page, 1280px viewport | [Home](home.jpg) |
| New Python getting-started guide, 1280px viewport | [Desktop](desktop.jpg) |
| New Python getting-started guide, 390px viewport | [Mobile](mobile.jpg) |

Browser checks: desktop and mobile navigation; keyboard search with ArrowDown
and Enter; search from a nested API page to a specific option anchor; Escape
closing the mobile drawer and returning focus; hidden closed mobile navigation;
copy buttons with actual copied Python text; horizontally scrolling option
and code tables without page overflow; shared API theme and direct class anchors.
The builder verifies every local file and fragment link and rejects duplicate IDs.

User pages contain current usage rather than release-preparation history. The
complete options catalog is checked against all 74 dataclass fields and accepted
loader aliases; tests require matching source docstrings and execute the guides
and README against actual build, export, checkpoint and exploration operations.
