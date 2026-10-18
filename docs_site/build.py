# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Mario Andrés Muñoz
"""Build the curated reference site and its consistently themed generated API.

The option catalog supplies descriptions, while the live dataclasses supply
field names, defaults and types. JSON aliases are read from the loader's AST.
A build fails if the catalog or generated links disagree with that source.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shutil
import tomllib
from dataclasses import fields
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

import markdown2
import pdoc
import pdoc.render

from instancespace.data.options import InstanceSpaceOptions

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs_site"
TOC = (
    (
        "Getting Started",
        (
            ("getting-started", "Getting Started"),
            ("metadata-format", "Metadata Format"),
            ("workflows", "Build and Explore"),
        ),
    ),
    (
        "Class Interface",
        (("instance-space", "InstanceSpace"), ("model", "Model and Persistence")),
    ),
    (
        "Pipeline Stages",
        (
            ("preprocessing", "PREPROCESSING"),
            ("prelim", "PRELIM"),
            ("sifted", "SIFTED"),
            ("pilot", "PILOT"),
            ("pythia", "PYTHIA"),
            ("cloister", "CLOISTER"),
            ("trace", "TRACE"),
        ),
    ),
    (
        "Topics",
        (
            ("options-reference", "Options Reference"),
            ("tuning", "Classifiers and Tuning"),
            ("outputs", "Outputs and Plotting"),
            ("plugins", "Custom Stages"),
            ("matlab-compatibility", "MATLAB Compatibility"),
            ("troubleshooting", "Troubleshooting"),
        ),
    ),
    ("API Reference", (("api-reference", "Modules and Signatures"),)),
)
ALIAS_POSITION = 2
CATALOG = cast(
    dict[str, dict[str, str]],
    json.loads((DOCS / "options.json").read_text()),
)
VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["poetry"]["version"],
)


def slug(value: str) -> str:
    """Make stable option and section anchors."""
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def option_aliases() -> dict[str, dict[str, str]]:
    """Read explicit accepted JSON aliases from the real loader."""
    tree = ast.parse((ROOT / "instancespace/data/options.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "InstanceSpaceOptions"
    )
    loader = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_dict"
    )
    ret = next(n for n in loader.body if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Call)
    mappings: dict[str, dict[str, str]] = {}
    for group in ret.value.keywords:
        assert group.arg is not None
        assert isinstance(group.value, ast.Call)
        call = group.value
        mapping = next(
            (k.value for k in call.keywords if k.arg == "field_mapping"),
            None,
        )
        if mapping is None and len(call.args) > ALIAS_POSITION:
            mapping = call.args[ALIAS_POSITION]
        mappings[group.arg] = {} if mapping is None else ast.literal_eval(mapping)
    return mappings


def check_catalog() -> int:
    """Require descriptions for every option and reject stale entries."""
    options = InstanceSpaceOptions.default()
    groups = {f.name for f in fields(options)}
    if groups != CATALOG.keys():
        raise ValueError(
            f"Option group documentation mismatch: {groups ^ CATALOG.keys()}",
        )
    count = 0
    for group in fields(options):
        expected = {f.name for f in fields(getattr(options, group.name))}
        actual = CATALOG[group.name].keys()
        if expected != actual:
            raise ValueError(
                f"Option documentation mismatch in {group.name}: {expected ^ actual}",
            )
        if any(not value.strip() for value in CATALOG[group.name].values()):
            raise ValueError(f"Missing substantive option description: {group.name}")
        count += len(expected)
    return count


def nav_html(prefix: str = "", current: str = "index") -> str:
    """Render one shared navigation for guides and API pages."""
    parts = [
        '<nav class="toc" id="doc-nav" aria-label="Contents">',
        f'<a class="toc-home" href="{prefix}index.html">Instance Space Analysis</a>',
        "<ul>",
    ]
    for section, items in TOC:
        opened = " open" if any(name == current for name, _ in items) else ""
        parts.append(f"<li><details{opened}><summary>{section}</summary><ul>")
        for name, title in items:
            active = ' class="current" aria-current="page"' if name == current else ""
            parts.append(f'<li><a{active} href="{prefix}{name}.html">{title}</a></li>')
        parts.append("</ul></details></li>")
    return "\n".join([*parts, "</ul></nav>"])


def header_html(prefix: str = "") -> str:
    """Render the searchable MATLAB-style reference header."""
    return (
        (DOCS / "templates/header.html")
        .read_text()
        .format(
            prefix=prefix,
            version=VERSION,
        )
    )


def footer_html() -> str:
    """Show current source and license information without release history."""
    return (DOCS / "templates/footer.html").read_text()


def page_html(
    title: str,
    purpose: str,
    body: str,
    stem: str,
    headings: list[tuple[str, str]],
) -> str:
    """Wrap a reference page in the shared accessible layout."""
    onpage = "".join(
        f'<li><a href="#{anchor}">{html.escape(label)}</a></li>'
        for anchor, label in headings
    )
    return (
        (DOCS / "templates/page.html")
        .read_text()
        .format(
            title=html.escape(title),
            description=html.escape(purpose, quote=True),
            header=header_html(),
            nav=nav_html(current=stem),
            purpose=html.escape(purpose),
            body=body,
            footer=footer_html(),
            onpage=onpage,
        )
    )


def option_table() -> str:
    """Render defaults and accepted aliases from code, with authored meanings."""
    options = InstanceSpaceOptions.default()
    aliases = option_aliases()
    parts = []
    for group in fields(options):
        value = getattr(options, group.name)
        parts.append(
            f"## options.{group.name}\n\n"
            "| Field | Default | Description |\n|---|---|---|",
        )
        for option in fields(value):
            key = f"{group.name}.{option.name}"
            default = repr(getattr(value, option.name))
            names = sorted(
                k
                for k, v in aliases[group.name].items()
                if v == option.name and k != option.name
            )
            description = html.escape(CATALOG[group.name][option.name]).replace(
                "_",
                "&#95;",
            )
            if names:
                description += (
                    "<br><small>JSON aliases: "
                    + ", ".join(f"<code>{html.escape(n)}</code>" for n in names)
                    + ".</small>"
                )
            parts.append(
                f'| <span id="opts-{slug(key)}"></span>`{option.name}` | '
                f"`{default}` | {description} |",
            )
        parts.append("")
    parts += [
        "## JSON names",
        "",
        (
            "Python field names and explicit JSON aliases in the tables are "
            "accepted case-insensitively. Duplicate aliases for the same "
            "field are rejected. Unknown fields are rejected. The "
            "`pilot.ISA3D` input selects a 2D/3D projection when `dims` is "
            "absent; an explicit `dims` must agree with it. "
            "`pythia.uselibsvm` is accepted and ignored. It does not select "
            "a classifier or tuning method."
        ),
        "",
        (
            "The normal Python configuration groups are shown above. MATLAB "
            "settings under `general.parallel` and `general.ncores` "
            "correspond to Python's `parallel.flag` and `parallel.n_cores`; "
            "they are not additional Python fields."
        ),
    ]
    return "\n".join(parts)


def render_markdown(text: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Render structured Markdown sections with stable local navigation."""
    lines = text.splitlines()
    title = lines[0].removeprefix("# ").strip()
    purpose = lines[2].strip()
    body = str(
        markdown2.markdown(
            "\n".join(lines[3:]),
            extras=["fenced-code-blocks", "tables", "header-ids", "markdown-in-html"],
        ),
    )
    headings: list[tuple[str, str]] = []

    def heading(match: re.Match[str]) -> str:
        label = html.unescape(re.sub("<[^>]+>", "", match[1]))
        anchor = slug(label)
        headings.append((anchor, label))
        return f'<h2 id="{anchor}">{match[1]}</h2>'

    body = re.sub(r'<h2(?: id="[^"]*")?>(.*?)</h2>', heading, body)
    chunks = re.split(r'(<h2 id="[^"]*">.*?</h2>)', body)
    if len(chunks) > 1:
        body = chunks[0] + "".join(
            f'<section class="refsect">{chunks[i]}{chunks[i+1]}</section>'
            for i in range(1, len(chunks), 2)
        )
    return title, purpose, body, headings


class HTMLInventory(HTMLParser):
    """Collect link, anchor, search and structural data for generated pages."""

    def __init__(self) -> None:
        """Initialize page inventories."""
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicates: set[str] = set()
        self.links: list[str] = []
        self.words: list[str] = []
        self.title = ""
        self.in_title = False
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Read IDs, resources and hidden script/style boundaries."""
        values = dict(attrs)
        if "id" in values:
            anchor = values["id"] or ""
            if anchor in self.ids:
                self.duplicates.add(anchor)
            self.ids.add(anchor)
        if tag in ("a", "link", "script", "img"):
            link = values.get("href") or values.get("src")
            if link:
                self.links.append(link)
        if tag == "title":
            self.in_title = True
        if tag in ("style", "script"):
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        """Close title and hidden content boundaries."""
        if tag == "title":
            self.in_title = False
        if tag in ("style", "script"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data: str) -> None:
        """Collect searchable visible text."""
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.words.append(data)


def inventory(output: Path) -> dict[Path, HTMLInventory]:
    """Parse every built page once."""
    pages: dict[Path, HTMLInventory] = {}
    for path in output.rglob("*.html"):
        parser = HTMLInventory()
        parser.feed(path.read_text())
        pages[path.resolve()] = parser
    return pages


def check_links(output: Path) -> int:
    """Fail on broken local files, fragments and duplicate IDs."""
    pages = inventory(output)
    errors: list[str] = []
    for path, page in pages.items():
        if page.duplicates:
            errors.append(f"{path}: duplicate IDs {page.duplicates}")
        for link in page.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or link.startswith("mailto:"):
                continue
            target = (
                (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            )
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{path.relative_to(output.resolve())}: missing {link}")
            elif (
                parsed.fragment
                and target in pages
                and unquote(parsed.fragment) not in pages[target].ids
            ):
                errors.append(
                    f"{path.relative_to(output.resolve())}: missing fragment {link}",
                )
    if errors:
        raise ValueError("\n".join(errors))
    return len(pages)


def search_index(output: Path) -> None:
    """Index curated pages, option fields and generated API references."""
    entries = []
    for path, page in inventory(output).items():
        entries.append(
            {
                "t": page.title.split(" — ")[0],
                "u": path.relative_to(output.resolve()).as_posix(),
                "p": "",
                "h": [],
                "x": " ".join(page.words).replace("_", " "),
            },
        )
    for group, descriptions in CATALOG.items():
        for field, description in descriptions.items():
            entries.append(
                {
                    "t": f"{group}.{field}",
                    "u": f'options-reference.html#opts-{slug(f"{group}.{field}")}',
                    "p": description,
                    "h": [field.replace("_", " ")],
                    "x": description,
                },
            )
    (output / "assets/search-index.js").write_text(
        "window.ISA_SEARCH = " + json.dumps(entries) + ";\n",
    )


def namespace_docstring(value: str, namespace: str) -> str:
    """Scope pdoc's repeated heading and footnote IDs to their API member."""
    value = str(value)
    anchors = set(re.findall(r'id="([^"]+)"', value))
    prefix = namespace or "module"
    for anchor in sorted(anchors, key=len, reverse=True):
        replacement = f"{prefix}-{anchor}"
        value = value.replace(
            f'id="{anchor}"',
            f'id="{replacement}" data-original-id="{anchor}"',
        )
        value = value.replace(f'href="#{anchor}"', f'href="#{replacement}"')
    return value


def api_reference(output: Path) -> str:
    """Build accurate API signatures with the same site chrome."""
    pdoc.render.configure(
        docformat="numpy",
        template_directory=DOCS / "pdoc-template",
        show_source=False,
        search=False,
    )
    pdoc.render.env.filters["isa_namespace"] = namespace_docstring
    pdoc.render.env.globals.update(
        isa_nav=nav_html,
        isa_header=header_html,
        isa_footer=footer_html,
        isa_version=VERSION,
    )
    pdoc.pdoc(
        "instancespace",
        "instancespace.model",
        "instancespace.plotting",
        "instancespace.utils.get_classifier_fcn",
        output_directory=output / "api",
    )
    for page in (output / "api").rglob("*.html"):
        parser = HTMLInventory()
        source = page.read_text()
        # typing.NamedTuple is a runtime factory; its repr contains an address.
        # Render its qualified name without unstable process-specific metadata.
        source = re.sub(
            r"&lt;function NamedTuple at 0x[0-9a-f]+&gt;",
            "typing.NamedTuple",
            source,
        )
        parser.feed(source)
        seen = set(parser.ids)

        def legacy_alias(match: re.Match[str]) -> str:
            anchor = match[1]
            if anchor in seen:
                return match[0]
            seen.add(anchor)
            return (
                match[0] + f'<span id="{anchor}" class="legacy-anchor" '
                'aria-hidden="true"></span>'
            )

        page.write_text(
            re.sub(r'<[^>]+data-original-id="([^"]+)"[^>]*>', legacy_alias, source),
        )
    (output / "api" / "index.html").write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=../api-reference.html">'
        "<title>API Reference</title></head><body>"
        '<a href="../api-reference.html">Open the API Reference</a></body></html>',
    )
    lines = [
        "# API Reference",
        "",
        "Modules, classes and signatures generated from the installed source",
        "",
        (
            "Use the curated stage and workflow pages for examples and "
            "interpretation. The modules below provide exact call "
            "signatures, parameters, return fields and configuration "
            "dataclasses."
        ),
        "",
        "| Module | Reference |",
        "|---|---|",
    ]
    for path in sorted((output / "api").rglob("*.html")):
        name = (
            path.relative_to(output / "api")
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )
        if name == "index":
            continue
        lines.append(
            f"| `{name}` | [Open reference]"
            f'(api/{path.relative_to(output / "api").as_posix()}) |',
        )
    return "\n".join(lines)


def build(output: Path) -> None:
    """Build the site, validating schema coverage and all generated links."""
    count = check_catalog()
    if (
        output.resolve() != (ROOT / "site").resolve()
        and output.exists()
        and any(output.iterdir())
        and not (output / ".instancespace-docs-build").exists()
    ):
        raise ValueError(
            "Refusing to replace a nonempty directory not owned by this builder.",
        )
    if output.exists():
        shutil.rmtree(output)
    (output / "assets").mkdir(parents=True)
    (output / ".instancespace-docs-build").write_text(
        "InstanceSpace documentation build\n",
    )
    for asset in (DOCS / "assets").iterdir():
        shutil.copyfile(asset, output / "assets" / asset.name)
    api = api_reference(output)
    expected = {"index"} | {stem for _, items in TOC for stem, _ in items}
    sources = {p.stem: p.read_text() for p in (DOCS / "src").glob("*.md")}
    sources["api-reference"] = api
    if sources.keys() != expected:
        raise ValueError(f"Navigation/source mismatch: {sources.keys() ^ expected}")
    for stem, source_text in sources.items():
        text = source_text
        if stem == "options-reference":
            text += "\n\n" + option_table()
        title, purpose, body, headings = render_markdown(text)
        (output / f"{stem}.html").write_text(
            page_html(title, purpose, body, stem, headings),
        )
    search_index(output)
    pages = check_links(output)
    print(
        f"Built {pages} pages; {count} option fields covered; "
        "local links and anchors pass.",
    )


def main() -> None:
    """Build or check an existing documentation tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(
            f"Checked {check_links(args.output)} pages; "
            f"{check_catalog()} options covered.",
        )
    else:
        build(args.output)


if __name__ == "__main__":
    main()
