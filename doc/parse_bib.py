import re, warnings
from util import el

def parse_bib(text):
    # TODO add other variations
    mappings = {
        r"\"{a}": "ä",
        r"\"{o}": "ö",
        r"\"{u}": "ü",
        r"\"{e}": "ë",
        r"\^{i}": "î",
        r"\'{e}": "é",
        r"\`{o}": "ò",
        r"\'{o}": "ó",
        r"\^{o}": "ô",
        r"\H{o}": "ő",
        r"\~{o}": "õ",
        r"\c{c}": "ç",
        r"\k{a}": "ą",
        r"\l{}": "ł",
        r"\={o}": "ō",
        r"\.{o}": "ȯ",
        r"\r{a}": "å",
        r"\u{o}": "ŏ",
        r"\v{s}": "š",
        r"\o{}": "ø",
        r"{\i}": "ı",
        "---": "—",
    }

    references = {}

    i = 0
    while i < len(text):
        i = text.find("@", i)

        if i == -1: break

        j = text.find("\n", i)

        reference_name = text[i:j].split("{")[-1].rstrip(",")

        start = j
        end = text.find("\n}", j)

        d = {}
        for line in text[start:end].strip().split("\n"):
            name, value = line.strip().split("=", 1)
            value = value.strip()

            if value.startswith("{"):
                if value.endswith("}"):
                    value = value[1:-1]
                elif value.endswith("},"):
                    value = value[1:-2]
                else:
                    raise ValueError("Expected that %s ends with either '}' or '},' since it started with '{'." % value)
            elif value.startswith('"'):
                if value.endswith('"'):
                    value = value[1:-1]
                elif value.endswith('",'):
                    value = value[1:-2]
                else:
                    raise ValueError("Expected that %s ends with either \" or \", since it started with \"." % value)

            for old, new in mappings.items():
                value = value.replace(old, new)
                old = old.replace("{", "").replace("}", "")
                value = value.replace(old, new)

            def replace(match):
                return match.group(1)[1:-1]

            value = re.sub("({.*?})", replace, value)

            d[name.strip()] = value

        authors = d["author"]
        referenceid = ""
        new_authors = []
        for author in authors.split(" and "):
            if ", " in author:
                last, first = author.split(", ")
                author = first + " " + last
            new_authors.append(author)
            referenceid += author.split()[-1][0]

        if len(new_authors) > 1:
            new_authors[-1] = "and " + new_authors[-1]

        if len(new_authors) != 2:
            authors = ", ".join(new_authors)
        else:
            authors = " ".join(new_authors)

        title = d["title"]

        parts = [authors, title]

        is_book = False

        if "journal" in d:
            journal = d["journal"]
            parts.append(el("i", journal))
        elif "booktitle" in d:
            booktitle = d["booktitle"]
            is_book = True
            parts.append(["In ", el("i", booktitle)])
        else:
            # TODO parse other formats
            #warnings.warn("WARNING: Parsing not yet implemented for: " + str(d))
            pass

        which = ""
        if "volume" in d:
            if "number" in d:
                which += d["volume"] + "(" + d["number"] + ")"
            else:
                which = "volume " + d["volume"]

        if "pages" in d:
            first, last = d["pages"].replace("--", "-").replace("–", "-").split("-")
            if which:
                which += ":"
            which += first + "–" + last

        if "year" in d:
            if which:
                which += ", "
            which += d["year"]

            referenceid += str(d["year"])[-2:]

        # TODO fix commas/points
        if which:
            if is_book:
                parts[-1] += ", " + which
            else:
                parts.append(which)

        parts_with_dot = []
        for part in parts:
            parts_with_dot.append(part)
            parts_with_dot.append(". ")

        references[reference_name] = {
            "referenceid": referenceid,
            "value": parts_with_dot,
            "referenced": False
        }

        i = end

    return references

def main():
    path = "pymatting/doc/source/pymatting.bib"

    with open(path, encoding="utf-8") as f:
        text = f.read()

    references = parse_bib(text)
    print(references)

if __name__ == "__main__":
    main()
