from pathlib import Path
from html import escape

def isiterable(x):
    try:
        iter(x)
        return True
    except Exception:
        return False

def flatten(x):
    if isinstance(x, str):
        yield x

    elif isiterable(x):
        for value in x:
            yield from flatten(value)

    else:
        yield x

class HTML:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"HTML({self.value})"

    __repr__ = __str__

def el(elementname, children=None, cls=None, close=True, **attributes):
    if cls is not None:
        attributes["class"] = cls
    attribs = "".join(f' {escape(key)}="{escape(value)}"'
        for key, value in attributes.items())
    result = [HTML(f"<{escape(elementname)}{attribs}>")]
    if children is not None:
        result.append(children)
        assert close
    if close:
        result.append(HTML(f"</{escape(elementname)}>"))
    return result

def li(children, **kwargs):
    return el("li", children, **kwargs)

def ol(children, **kwargs):
    return el("ol", children, **kwargs)

def ul(children, **kwargs):
    return el("ul", children, **kwargs)

def span(children, **kwargs):
    return el("span", children, **kwargs)

def div(children, **kwargs):
    return el("div", children, **kwargs)

def a_link(url, children, **kwargs):
    return el("a", children, href=url, **kwargs)

def img(src, **kwargs):
    return el("img", src=src, close=False, **kwargs)

def read_text(filename):
    return Path(filename).read_text(encoding="utf-8")

