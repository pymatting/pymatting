import re, ast
from highlight import escape

class Stream:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def peek(self, n=1):
        return self.text[self.pos:self.pos + n]

    def consume(self, n=1):
        result = self.peek(n)
        self.skip(n)
        return result

    def available(self):
        return len(self.text) - self.pos

    def skip(self, n=1):
        assert self.available() >= n
        self.pos += n

    def __bool__(self):
        return self.available() > 0

    def match(self, pattern, flags=0):
        r = re.compile(pattern, flags=flags)
        return r.match(self.text, pos=self.pos)

    def match_consume(self, pattern, flags=0):
        if m := self.match(pattern, flags):
            self.pos = m.end()
        return m

def parse_whitespace(s):
    indentation = 0
    while s:
        if s.peek() == " ":
            s.skip()
            indentation += 1
        else:
            return indentation

def is_whitespace(s):
    return len(s.strip()) == 0

def skip_whitespace(stream):
    while stream and is_whitespace(stream.peek()):
        stream.skip()

def skip_empty_lines(stream):
    while stream and stream.match_consume(" *(\n|$)"):
        pass

def make_node(node_type, node_value, **kwargs):
    return {"type": node_type, "value": node_value, **kwargs}

name_pattern = "[a-zA-Z_][a-zA_Z_0-9]*"

url_char = "[a-zA-Z0-9" + escape("!#$%&'()*+,-./:;=?@[]_~") + "]"
url_pattern = "(:?https?:\/\/|www.)" + url_char + "+"

def parse_text(stream, end="\n\n"):
    skip_whitespace(stream)

    parts = []

    text = ""
    while stream and stream.peek(len(end)) != end:
        node = None

        if m := stream.match_consume(r"\:(?P<name>[a-zA-Z_]+)\:`(?P<value>.*?)`"):
            node = make_node("inline_" + m.group("name"), m.group("value"))

        elif m := stream.match_consume(r"`(?P<code>.*?)`"):
            node = make_node("inline_code", m.group("code"))

        elif stream.peek(2) == "![":
            assert stream.consume(2) == "!["
            alt = parse_text(stream, end="](")
            url = stream.match_consume(r"\]\((.*?)\)").group(1)
            node = make_node("image", url, alt=alt)

        elif stream.match("\\[[^\n]+?\\]\\("):
            assert stream.consume(1) == "["
            value = parse_text(stream, end="](")
            url = stream.match_consume(r"\]\((.*?)\)").group(1)

            node = make_node("url", value, url=url)

        elif m := stream.match_consume(url_pattern):
            url = m.group(0)
            node = make_node("url", url, url=url)

        # If we found something different from text
        if node:
            # Flush text
            if text:
                parts.append(make_node("text", text))
                text = ""
            # Append different kind of node
            parts.append(node)
        else:
            text += stream.consume()

    # Flush text
    if text:
        parts.append(make_node("text", text))

    if len(parts) == 1:
        return parts[0]
    else:
        return make_node("blocks", parts)

def is_indented(stream):
    return stream.match(" +")

def parse_codeblock(stream):
    indentation = len(stream.match(" *").group(0))

    codeblock = []
    while True:
        # Consume lines if they are whitespace
        while m := stream.match_consume(" *\n"):
            codeblock.append(m.group(0).rstrip("\n"))

        if not is_indented(stream): break

        # Consume indented line
        line = parse_raw_line(stream)
        assert is_whitespace(line[:indentation]), f"Line should have indentation depth of {indentation}"
        codeblock.append(line[indentation:])

    return make_node("code_block", "\n".join(codeblock))

def parse_unordered_list(stream):
    items = []
    while stream.peek() == "*":
        stream.skip()
        items.append(parse_text(stream, end="\n"))
        if stream.peek() == "\n":
            stream.skip()
        else:
            break
    return make_node("ul", items)

def is_ordered_list(stream):
    return stream.match("[0-9]+\\.")

def parse_ordered_list(stream):
    items = []
    while is_ordered_list(stream):
        while stream.peek() in "0123456789":
            stream.skip()
        assert stream.consume() == "."
        items.append(parse_text(stream, end="\n"))
        if stream.peek() == "\n":
            stream.skip()
        else:
            break
    return make_node("ol", items)

def parse_raw_line(stream):
    return stream.match_consume("(.*?)(\n|$)").group(1)

underlined_pattern = re.compile("(.*?)\n[+-=]+\n")

def is_underlined_block(stream):
    return underlined_pattern.match(stream.text, pos=stream.pos)

def parse_underlined_block(stream):
    header = parse_raw_line(stream)
    dividers = parse_raw_line(stream)

    if len(dividers) != len(header):
        print(f"WARNING: Incorrect number of dividers:\n{header}\n{dividers}\n")

    if m := stream.match_consume(">>> (.+\n?)+"):
        code_block = make_node("code_block", m.group(0))
        return make_node("block", code_block, header=header)

    parameters = []
    while stream:
        if m := stream.match_consume("(?P<parameter>[a-zA-Z_][a-zA-Z_0-9]*) *: *(?P<details>.*)\n"):

            description = []
            while is_indented(stream):
                description.append(parse_text(stream, end="\n"))
                if stream.peek() == "\n":
                    stream.skip()

            # Insert space between text nodes
            # TODO Find a less hacky way to do this.
            for node in description[:-1]:
                if node["type"] == "text":
                    node["value"] += " "
                elif node["type"] == "blocks":
                    node["value"].append(make_node("text", " "))
                else:
                    raise ValueError(f"Expected parse_text to return text or blocks node, but got {node} instead")

            if len(description) == 1:
                description = description[0]
            else:
                description = make_node("blocks", description)

            parameter = {
                "parameter": m.group("parameter"),
                "details": m.group("details"),
                "description": description,
            }

            parameters.append(parameter)
        else:
            break

    return make_node("parameters", parameters, header=header)

def parse_stream(stream):
    blocks = []

    while True:
        skip_empty_lines(stream)

        if not stream: break

        if m := stream.match_consume("#+"):
            n = len(m.group(0))
            blocks.append(make_node("h" + str(n), parse_text(stream)))

        elif stream.peek() == "*":
            blocks.append(parse_unordered_list(stream))

        elif is_indented(stream):
            blocks.append(parse_codeblock(stream))

        elif is_ordered_list(stream):
            blocks.append(parse_ordered_list(stream))

        elif m := stream.match_consume(".. (?P<name>[a-z]+)::( *\n)*(?P<block>(.+\n)+)"):
            blocks.append(make_node(m.group("name") + "_block", m.group("block")))

        elif is_underlined_block(stream):
            blocks.append(parse_underlined_block(stream))

        else:
            text = parse_text(stream)
            blocks.append(make_node("text_block", text))

    if len(blocks) == 1:
        return blocks[0]

    return make_node("blocks", blocks)

def parse(text):
    stream = Stream(text)
    return parse_stream(stream)

def node_text(node, lines):
    if node.lineno == node.end_lineno:
        return lines[node.lineno - 1][node.col_offset:node.end_col_offset]
    else:
        a = lines[node.lineno - 1][node.col_offset:]
        b = lines[node.end_lineno - 1][:node.end_col_offset]
        between = lines[node.lineno:node.end_lineno - 1]
        return "\n".join([a] + between + [b])

def parse_python_file(filename):
    with open(filename, encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code)

    lines = code.split("\n")

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    functions = []
    for i, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)

            funcname = node.name

            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
            args = []
            for arg, default in zip(node.args.args, defaults):
                args.append((arg.arg, node_text(default, lines) if default else None))

            parents = []
            if node.parent and isinstance(node.parent, ast.ClassDef):
                parents.append(node.parent.name)

            parsed_docstring = parse(docstring) if docstring is not None else None

            functions.append(make_node(
                "function",
                parsed_docstring,
                funcname=funcname,
                parents=parents,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                filename=str(filename),
                args=args))

    return make_node("file", functions, filename=str(filename))

def compare(source, expected):
    result = parse(source)

    assert result == expected, f"Expected:\n{expected}\nResult:\n{result}"

def test_heading():
    source = "# foo"

    expected = {
        "type": "h1",
        "value": {
            "type": "text",
            "value": "foo",
        }
    }

    compare(source, expected)

    source = "## bar"

    expected = {
        "type": "h2",
        "value": {
            "type": "text",
            "value": "bar",
        }
    }

    compare(source, expected)

def test_unordered_list():
    source = """
* this is
* an unordered list
"""

    expected = {
        "type": "ul",
        "value": [
            {"type": "text", "value": "this is"},
            {"type": "text", "value": "an unordered list"},
        ],
    }

    compare(source, expected)

def test_indented_code_block():
    source = """
    indented
    code
    block
"""

    expected = {
        "type": "code_block",
        "value": "indented\ncode\nblock",
    }

    compare(source, expected)

def test_ordered_list():
    source = """

1. this
1. is
123. an ordered list

    """

    expected = {
        "type": "ol",
        "value": [
            {"type": "text", "value": "this"},
            {"type": "text", "value": "is"},
            {"type": "text", "value": "an ordered list"},
        ],
    }

    compare(source, expected)

def test_math_block():
    source = r"""
.. math::

    2^n
"""
    expected = {
        "type": "math_block",
        "value": "    2^n\n"
    }

    compare(source, expected)

def test_parameters():
    source = """

Parameters
==========
foo: 123
    a description
bar: 456
    another description

"""
    expected = {
        'type': 'parameters',
        'value': [
            {'parameter': 'foo', 'details': '123', 'description': {'type': 'text', 'value': 'a description'}},
            {'parameter': 'bar', 'details': '456', 'description': {'type': 'text', 'value': 'another description'}},
        ],
        'header': 'Parameters',
    }

    compare(source, expected)

def test_inline_code():
    source = "some text `foo()` more text"

    expected = {
        "type": "text_block",
        "value": {
            "type": "blocks",
            "value": [
                {"type": "text", "value": "some text "},
                {"type": "inline_code", "value": "foo()"},
                {"type": "text", "value": " more text"},
            ],
        },
    }

    compare(source, expected)

def test_inline_math():
    source = ":math:`h \\times w`"

    expected = {
        "type": "text_block",
        "value": {
            "type": "inline_math",
            "value": "h \\times w",
        }
    }

    compare(source, expected)

def test_image():
    source = "an image ![Image of a lemur](lemur_small.png) followed by more text"

    expected = {
        "type": "text_block",
        "value": {
            "type": "blocks",
            "value": [
                {"type": "text", "value": "an image "},
                {"type": "image", "value": "lemur_small.png", "alt": {"type": "text", "value": "Image of a lemur"}},
                {"type": "text", "value": " followed by more text"},
            ],
        },
    }

    compare(source, expected)

def test_block():
    source = """
Example
=======
>>> (1 +
...      2)
3
"""

    expected = {
        'type': 'block',
        'header': 'Example',
        'value': {
            "type": "code_block",
            "value": ">>> (1 +\n...      2)\n3\n",
        },
    }

    compare(source, expected)

def test_https():
    source = "https://www.example.com#foo"

    expected = {
        'type': 'text_block',
        'value': {
            "type": "url",
            "value": source,
            "url": "https://www.example.com#foo",
        },
    }

    compare(source, expected)

def test_url():
    source = "[`foo()`](example.com)"

    expected = {
        'type': 'text_block',
        'value': {
            "type": "url",
            "value": {
                "type": "inline_code",
                "value": "foo()",
            },
            "url": "example.com",
        },
    }

    compare(source, expected)

def test_stream_match():
    s = Stream("Hello, World!")

    assert s.match("[a-zA-Z]+").group(0) == "Hello"
    assert s.consume(7) == "Hello, "
    assert s.match("[a-zA-Z]+").group(0) == "World"

def run_all_tests():
    import traceback

    num_passed_tests = 0
    num_tests = 0
    for name, func in globals().items():
        if name.startswith("test_") and callable(func):
            num_tests += 1
            try:
                func()
                num_passed_tests += 1
            except Exception as e:
                traceback.print_exc()
                print("\nTest", name, "failed\n")

    print("Passed", num_passed_tests, "of", num_tests, "tests.")

if __name__ == "__main__":
    run_all_tests()
