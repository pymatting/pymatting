import re, string, html, keyword

def group(x):
    return "(" + x + ")"

def non_capturing_group(x):
    return "(?:" + x + ")"

def named_group(name, x):
    return "(?P<" + name + ">" + x + ")"

def opt(x):
    return non_capturing_group(x) + "?"

def any_of(*args):
    return non_capturing_group("|".join(non_capturing_group(x) for x in args))

def escape(x):
    return "".join("\\" + c for c in x)

# 1. int or frac part is always mandatory, but exp part is optional
# 2. need to try to match int + frac first, then try int and frac individually
ipart = "[0-9]+"
fpart = r"\.[0-9]+"
epart = "[eEgG][+-]?[0-9]+"

strings = [
    '""".*?"""',
    "'''.*?'''",
    r"'.*?'",
    '".*?"',
]

prefixed_strings = []
for s in strings:
    for prefix in ["", "r", "f"]:
        prefixed_strings.append(prefix + s)

keywords = set("""
abs all any ascii bin bool bytearray bytes callable chr classmethod compile
complex delattr dict dir divmod enumerate eval exec filter float format
frozenset getattr globals hasattr hash help hex id input int isinstance
issubclass iter len list locals map max memoryview min next object oct open ord
pow print property range repr reversed round set setattr slice sorted
staticmethod str sum super tuple type vars zip""".strip().split() + keyword.kwlist)

patterns = {
    "hex": "0x" + "[0-9a-f]+",
    "indentation": any_of("^>>> ", r"^\.\.\. "),
    "number": any_of(ipart + fpart, ipart, fpart) + opt(epart),
    "name": "[a-zA-Z_]" + "[a-zA-Z_0-9]*",
    "space": "[ \r\n\t]+",
    "comment": "#.*?\n",
    "string": any_of(*prefixed_strings),
    "operator": "[" + "".join(escape(c) for c in string.punctuation) + "]",
}

pattern = "|".join(group(pattern) for pattern in patterns.values())

codestyle = """
<style>

</style>
"""

def indentation(line):
    return len(line) - len(line.lstrip())

def remove_too_much_identation(code):
    lines = code.split("\n")
    min_indent = min(indentation(line) for line in lines if line.strip())
    lines = [line[min_indent:] for line in lines]
    return "\n".join(lines)

def highlight(code, output):
    for match in re.finditer(pattern, code, flags=re.DOTALL | re.MULTILINE):
        for name, value in zip(patterns, match.groups()):
            if value is not None:
                start, end = match.span()
                if value in keywords: name = "keyword"
                output.append("<span class=" + name + ">" + html.escape(value) + "</span>")

def highlight_inline(code):
    output = ['<span class="codeinline">']
    highlight(code, output)
    output.append("</span>")
    return "".join(output)


def highlight_block(code):
    output = ['<div class="codeblock">']
    code = remove_too_much_identation(code)
    highlight(code, output)
    output.append("</div>")
    return "".join(output)
'''
def main():
    testcode = """
pi = 3.14159
r = 1.0
area = pi * r^2
arr[0x123]
# comment
print('str1' + "str2")
""".strip()

    with open(__file__) as f:
        testcode = f.read()

    begin_website = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TODO title</title>
</head>
<body>
</body>
"""

    end_website = """
</body>
</html>
"""

    with open("tmp.html", "w") as f:
        f.write(begin_website)
        f.write(codestyle)
        f.write(highlight(testcode))
        f.write(end_website)


if __name__ == "__main__":
    main()
'''
