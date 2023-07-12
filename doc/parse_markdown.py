import re, ast

class Stream:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def peek(self, n=1):
        return self.peek_at(0, n)

    def consume(self, n=1):
        result = self.peek(n)
        self.skip(n)
        return result

    def peek_at(self, offset, n=1):
        return self.text[self.pos + offset:self.pos + offset + n]

    def peek_until(self, end):
        s = ""
        n = 0
        while n <= self.available() and self.peek_at(n, len(end)) != end:
            s += self.peek_at(n)
            n += 1
        return s

    def consume_until(self, end):
        result = self.peek_until(end)
        self.skip(len(result))
        return result

    def available(self):
        return len(self.text) - self.pos

    def skip(self, n=1):
        assert self.available() >= n
        self.pos += n

    def __bool__(self):
        return self.available() > 0

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
    while stream:
        line = stream.peek_until("\n")

        # Skip line if it is whitespace
        if is_whitespace(line):
            stream.skip(len(line))
            if stream.peek() == "\n":
                stream.skip()
        else:
            break

def make_node(node_type, node_value, **kwargs):
    return {"type": node_type, "value": node_value, **kwargs}

name_pattern = "[a-zA-Z_][a-zA_Z_0-9]*"

def parse_inline(stream):
    assert stream.consume() == ":"
    name = stream.consume_until(":")
    assert re.fullmatch(name, name)
    assert stream.consume(2) == ":`"
    value = stream.consume_until("`")
    assert stream.consume() == "`"
    return make_node("inline_" + name, value)

url_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'()*+,-./:;=?@[]_~"

def parse_text(stream, end="\n\n"):
    skip_whitespace(stream)

    parts = []

    text = ""
    while stream and stream.peek(len(end)) != end:
        node = None

        if re.match(":" + name_pattern + ":`", stream.peek(10)):
            # TODO restructure so text does not need to be flushed everywhere
            if text:
                parts.append(make_node("text", text))
                text = ""

            parts.append(parse_inline(stream))

        elif stream.peek() == "`":
            if text:
                parts.append(make_node("text", text))
                text = ""

            assert stream.consume() == "`"
            code = stream.consume_until("`")
            assert stream.consume() == "`"
            parts.append(make_node("inline_code", code))

        elif stream.peek(2) == "![":
            if text:
                parts.append(make_node("text", text))
                text = ""

            assert stream.consume(2) == "!["
            alt = parse_text(stream, end="](")
            assert stream.consume(2) == "]("
            url = stream.consume_until(")")
            assert stream.consume() == ")"

            image_node = make_node("image", url, alt=alt)

            parts.append(image_node)

        # TODO stream.match
        elif re.match("\\[[^\n]+?\\]\\(", stream.peek(100)):
            if text:
                parts.append(make_node("text", text))
                text = ""

            assert stream.consume(1) == "["
            value = parse_text(stream, end="](")
            assert stream.consume(2) == "]("
            url = stream.consume_until(")")
            assert stream.consume() == ")"

            node = make_node("url", value, url=url)

            parts.append(node)

        elif stream.peek(7) == "http://" or stream.peek(8) == "https://" or stream.peek(4) == "www.":
            if text:
                parts.append(make_node("text", text))
                text = ""

            url = ""
            while stream and stream.peek() in url_chars:
                url += stream.consume()
            parts.append(make_node("url", None, url=url))

        else:
            text += stream.consume()

    if text:
        parts.append(make_node("text", text))

    if len(parts) == 1:
        return parts[0]
    else:
        return make_node("blocks", parts)

def parse_line(stream):
    return parse_text(stream, end="\n")

def is_indented(stream):
    line = stream.peek_until("\n")
    return len(line) > len(line.lstrip())

def parse_codeblock(stream):
    line = stream.peek_until("\n")
    indentation = len(line) - len(line.lstrip())

    codeblock = []
    while True:
        # Consume lines if they are whitespace
        while stream:
            line = stream.peek_until("\n")

            if line.strip(): break

            stream.skip(len(line))

            if stream.peek() == "\n":
                stream.skip()
            codeblock.append(line)

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
        items.append(parse_line(stream))
        if stream.peek() == "\n":
            stream.skip()
        else:
            break
    return make_node("ul", items)

def is_ordered_list(stream):
    line = stream.peek_until("\n")
    return re.match("[0-9]+\\.", line)

def parse_ordered_list(stream):
    items = []
    while is_ordered_list(stream):
        while stream.peek() in "0123456789":
            stream.skip()
        assert stream.consume() == "."
        items.append(parse_line(stream))
        if stream.peek() == "\n":
            stream.skip()
        else:
            break
    return make_node("ol", items)

def parse_raw_line(stream):
    line = stream.consume_until("\n")
    if stream.peek() == "\n":
        stream.skip()
    return line

def parse_block(stream):
    assert stream.consume(3) == ".. "
    name = ""
    abc = "abcdefghijklmnopqrstuvwxyz"
    while stream.peek() in abc:
        name += stream.consume()
    assert stream.consume(2) == "::"
    skip_empty_lines(stream)
    block = stream.consume_until("\n\n")
    return make_node(name + "_block", block)

underlined_pattern = re.compile("(.*?)\n[+-=]+\n")

def is_underlined_block(stream):
    return underlined_pattern.match(stream.text, pos=stream.pos)

def parse_underlined_block(stream):
    header = parse_raw_line(stream)
    dividers = parse_raw_line(stream)

    if len(dividers) != len(header):
        print(f"WARNING: Incorrect number of dividers:\n{header}\n{dividers}\n")

    if stream.peek(4) == ">>> ":
        code = stream.consume_until("\n\n")
        code_block = make_node("code_block", code)
        return make_node("block", code_block, header=header)

    parameters = []
    while stream:
        line = stream.peek_until("\n")
        if re.match("[a-zA-Z_][a-zA-Z_0-9]* *:", line):
            line = parse_raw_line(stream)
            parameter, details = line.split(":", 1)
            parameter = parameter.strip()
            details = details.strip()

            description = []
            while is_indented(stream):
                description.append(parse_text(stream, end="\n"))
                if stream.peek() == "\n":
                    stream.skip()

            if len(description) == 1:
                description = description[0]
            else:
                description = make_node("blocks", description)

            parameter = {
                "parameter": parameter,
                "details": details,
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

        if stream.peek() == "#":
            n = 0
            while stream.peek() == "#":
                stream.skip()
                n += 1

            blocks.append(make_node("h" + str(n), parse_text(stream)))

        elif stream.peek() == "*":
            blocks.append(parse_unordered_list(stream))

        elif is_indented(stream):
            blocks.append(parse_codeblock(stream))

        elif is_ordered_list(stream):
            blocks.append(parse_ordered_list(stream))

        elif stream.peek(3) == ".. ":
            blocks.append(parse_block(stream))

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
    with open(filename, "r", encoding="utf-8") as f:
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
            {'parameter': 'bar', 'details': '456', 'description': {'type': 'text', 'value': 'another description'}}
        ],
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
    source = "https://www.example.com"

    expected = {
        'type': 'text_block',
        'value': {
            "type": "url",
            "value": None,
            "url": "https://www.example.com",
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
