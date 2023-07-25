import os, re, sys, json, time, shutil, parse_markdown, parse_bib, highlight, urllib.request
from util import HTML, el, li, ol, ul, span, div, a_link, img, read_text, flatten, escape
from pathlib import Path

src_dir = Path("../pymatting")

def generate_html(node, references, html=None):
    if html is None:
        html = []

    if node is None:
        return html

    if node["type"] == "blocks":
        for child in node["value"]:
            generate_html(child, references, html)

    elif node["type"] == "block":
        html.append(div([
            el("h4", node["header"]),
            generate_html(node["value"], references),
        ], cls="block"))

    elif node["type"] == "parameters":
        html.append(el("h4", node["header"])),

        parameters = []
        for parameter in node["value"]:
            parameters.append(
                li(cls="parameteritem", children=[
                    span(parameter["parameter"], cls="parameter"),
                    [" (", el("i", parameter["details"]), ")"] if parameter["details"] else "",
                    ul(li(
                        cls="parameterdescription",
                        children=generate_html(parameter["description"], references),
                    )),
                ])
            )
            description = parameter["description"]

        html.append(ul(parameters))

    elif node["type"] == "text":
        html.append(span(node["value"], cls="text"))

    elif node["type"] == "inline_cite":
        name = node["value"]
        reference = references[name]
        referenceid = reference["referenceid"]
        reference["referenced"] = True

        url = f"/references.html#{referenceid}"

        html.append(span(cls="citation", children=a_link(url, f"[{referenceid}]")))

    elif node["type"] == "inline_math":
        html.append(span(cls="inline_math", children=r"\(" + node["value"] + r"\)"))

    elif node["type"] == "math_block":
        html.append(div(cls="math_block", children=r"$$" + node["value"] + "$$"))

    elif node["type"] == "code_block":
        highlighted = highlight.highlight_block(node["value"].rstrip())
        html.append(div(cls="code_block", children=HTML(highlighted)))

    elif node["type"] == "html_block":
        html.append(HTML(node["value"]))

    elif node["type"] == "text_block":
        html.append(div(cls="textblock", children=generate_html(node["value"], references)))

    elif node["type"] == "inline_code":
        highlighted = highlight.highlight_inline(node["value"])
        html.append(span(cls="inline_code", children=HTML(highlighted)))

    elif node["type"] == "ul":
        items = [li(generate_html(item, references)) for item in node["value"]]
        html.append(ul(items))

    elif node["type"] == "ol":
        items = [li(generate_html(item, references)) for item in node["value"]]
        html.append(ol(items))

    elif node["type"] == "url":
        url = node["url"]
        if node["value"]:
            body = generate_html(node["value"], references)
        else:
            body = url
        html.append(a_link(url, body))

    elif node["type"] == "file":
        for child in node["value"]:
            generate_html(child, references, html)

    elif node["type"] == "image_block":
        src = node["value"].strip()
        html.append(img(src=src))

    elif node["type"] == "function":
        funcname = node["funcname"]

        if funcname.startswith("_") and not funcname.startswith("__") or not node["value"]:
            return html

        args = [
            f"{arg}={default}" if default else arg
            for arg, default in node["args"]]

        signature = f"{funcname}({', '.join(args)})"

        if len(signature) > 50:
            signature = funcname + "(\n    " + ",\n    ".join(args) + ")"

        signature = HTML(highlight.highlight_block(signature))

        prefix = str(Path(node["filename"]).relative_to(src_dir).parent).replace("/", ".") + "."
        for parent in reversed(node["parents"]):
            prefix += parent + "."

        fullname = prefix + funcname

        node["fullname"] = fullname

        source_url = str(Path("https://github.com/pymatting/pymatting/blob/master/pymatting") / Path(node["filename"]).relative_to(src_dir)) + "#L" + str(node["lineno"]) + "-L" + str(node["end_lineno"])

        html.append(div(div(cls="function", children=[
            div(id=funcname, children=[
                a_link(source_url, img("/figures/github-mark.svg", cls="github-icon", title="Function definition on GitHub")),
                el("h3", fullname, cls="functionname"),
                a_link("#" + funcname, " 🔗", cls="functionanchorlink", title="Permalink to this definition"),
            ]),
            div([
                el("h4", "Signature"),
                ul(li(signature, cls="parameterdescription")),
            ], cls="signature"),
            el("h4", "Function Description"),
            div(generate_html(node["value"], references)),
        ])))

    elif re.match("h[0-9]+", node["type"]):
        html.append(el(node["type"], generate_html(node["value"], references)))

    else:
        raise ValueError(f"Unrecognized node:\n{json.dumps(node, indent=4)[:200]}")

    return html

def main():
    directories = [
        ("alpha", "Alpha Estimation"),
        ("cutout", "Cutout Function"),
        ("foreground", "Foreground Estimation"),
        ("laplacian", "Matting Laplacians"),
        ("preconditioner", "Preconditioners"),
        ("solver", "Solvers"),
        ("util", "Utility Functions"),
    ]

    bib_path = "source/biblography.bib"

    build_dir = Path("build")

    try:
        shutil.rmtree(build_dir)
    except FileNotFoundError:
        pass

    year = time.strftime("%Y")

    footer = ", ".join([
        f"© Copyright {year}",
        "Thomas Germer",
        "Tobias Uelwer",
        "Stefan Conrad",
        "Stefan Harmeling",
    ])

    references = parse_bib.parse_bib(read_text(bib_path))

    page_infos = [
        ("PyMatting", "source/index.md", "index.html"),
        ("API Reference", None, None),
        ("Examples", "source/examples.md", "examples.html"),
        ("Benchmarks", "source/benchmarks.md", "benchmarks.html"),
        ("Biblography", None, None),
        ("PyPI", "https://pypi.org/project/PyMatting/", None),
        ("GitHub", "https://www.github.com/pymatting/pymatting", None),
    ]

    os.makedirs("static/fonts", exist_ok=True)

    # Download KaTeX
    for url in [
        "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css",
        "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js",
        "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js",
    ]:
        filename = os.path.join("static", url.split("/")[-1])
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)

    # Download KaTeX fonts
    with open("static/katex.min.css") as f:
        for font in re.findall(r"fonts/KaTeX_[a-zA-Z0-9\-]+.(?:(?:ttf)|(?:woff2)|(?:woff))", f.read()):
            filename = os.path.join("static", font)
            if not os.path.isfile(filename):
                url = "https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/" + font
                print("Downloading", url, "to", filename)
                urllib.request.urlretrieve(url, filename)

    shutil.copytree("static", "build")

    lines = read_text(src_dir / "__about__.py").split("\n")
    version_line = next(line for line in lines if "__version__" in line)
    version = version_line.split("=")[-1].strip('" \'')

    def write_website(html_path, title, content):
        head = [
            HTML('<meta charset="utf-8" />'),
            el("title", f"{title} — PyMatting {version} documentation"),
            el("link", rel="stylesheet", href="/katex.min.css", close=False),
            el("link", rel="stylesheet", href="/style.css", close=False),
            el("script", src="/katex.min.js"),
        ]

        api_reference = [
            li(a_link(f"/{sub_dir}.html", title2, cls="sidebarlink" + (" currentpage" if title == title2 else "")))
            for sub_dir, title2 in directories]

        pages = []
        for title2, src_path, url in page_infos:
            cls = "sidebarlink"
            if title == title2:
                cls += " currentpage"

            if title2 == "API Reference":
                pages.append(li([
                        a_link("/api.html", "API Reference", cls=cls),
                        HTML("<br>"),
                        ul(api_reference)], cls=cls))
            elif title2 == "Biblography":
                pages.append(li(a_link("/references.html", "Biblography", cls=cls)))
            elif src_path.startswith("https"):
                pages.append(li(a_link(src_path, title2, cls=cls)))
            else:
                pages.append(li(a_link("/" + url, title2, cls=cls)))

        sidebar = div(cls="sidebar", children=[
            div(cls="logo", children=[
                a_link("/", img("/figures/lemur_small.png", width="50px"), cls=cls),
                a_link("/", "PyMatting", cls="logotext"),
            ]),
            div("CONTENTS", cls="sidebarcontents"),
            ul(pages),
        ])

        middle = div(cls="middle", children=[
            el("h1", title),
            content,
            el("footer", footer),
        ])

        body = [sidebar, middle]

        # Run KaTeX when page has loaded
        body.append(el("script",
            src="/auto-render.min.js",
            onload="renderMathInElement(document.body)"))

        html = [
            HTML("<!DOCTYPE html>"),
            el("html", lang="en", children=[
                el("head", head),
                el("body", body),
            ])
        ]

        # Escape non-HTML elements (strings)
        escaped_strings = (element.value if isinstance(element, HTML) else escape(element)
            for element in flatten(html))

        html_path.parent.mkdir(exist_ok=True, parents=True)
        html_path.write_text("".join(escaped_strings), encoding="utf-8")

    for title, src_path, html_path in page_infos:
        if src_path is not None and html_path is not None:
            page = parse_markdown.parse(read_text(src_path))
            content = generate_html(page, references)
            write_website(build_dir / html_path, title, content)

    packages = []
    for directory, title in directories:
        html_path = build_dir / (directory + ".html")
        directory = src_dir/ directory

        content = []
        function_links = []
        for path in sorted(directory.rglob("*.py")):
            if path.name.startswith("__"): continue
            functions = parse_markdown.parse_python_file(path)
            generate_html(functions, references, content)

            for function in functions["value"]:
                if "fullname" in function:
                    url = str(path.relative_to(src_dir).parent.with_suffix(".html")) + "#" + function["funcname"]
                    name = function["fullname"]
                    function_links.append(li(a_link(url, name)))

        write_website(html_path, title, content)

        packages.append(li([
            title,
            ul(function_links),
        ]))

    write_website(build_dir / "api.html", "API References", ul(packages))

    reference_items = []
    for reference in sorted(references.values(), key=lambda r: r["referenceid"]):
        if reference["referenced"]:
            referenceid = reference["referenceid"]
            reference_items.append(li(cls="referenceitem", children=[
                a_link("#" + referenceid, referenceid, id=referenceid),
                span(ul(li(reference["value"]))),
            ]))
    write_website(build_dir / "references.html", "Biblography", ul(reference_items))

if __name__ == "__main__":
    main()
