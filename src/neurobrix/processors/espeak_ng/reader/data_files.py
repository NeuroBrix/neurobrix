"""Parsers of the vendor's documented text formats: the word list, the phoneme definitions, the voice files.

Written from the data files, their documentation (the dictionary, phoneme-table and voice pages) and observed
outputs only. Standard library only. Every refusal is a named error that says the file, the line and what is wrong.
"""
import os
import re


class DataFileError(ValueError):
    """A data file does not follow its documented format."""


def _fail(source, number, why):
    raise DataFileError("%s line %d: %s" % (source, number, why))


def strip_comment(line):
    """Cuts a `//` comment that starts the line or follows white space (a `//` inside a field is data)."""
    m = re.search(r"(^|\s)//", line)
    return line[: m.start()] if m else line


# ---------------------------------------------------------------------------------------------------------------
# The word list (`<language>_list`): <word> [<phoneme string>] [<flags>], optional `?N` / `?!N` condition first.

class ListEntry(object):
    __slots__ = ("words", "phonemes", "text", "flags", "condition", "line", "excess")

    def __init__(self, words, phonemes, text, flags, condition, line, excess=()):
        self.words = words          # tuple of the written words (one, or the members of a bracketed group)
        self.phonemes = phonemes    # the phoneme string, or None when the entry gives flags only
        self.text = text            # the replacement text of a `$text` / text-mode entry, or None
        self.flags = flags          # tuple of flag names without the `$`
        self.condition = condition  # None, or (number, wanted): the voice must (not) declare this dictrules number
        self.line = line
        self.excess = excess        # the fields beyond the format's, which no reading may use (see the reader)

    def __repr__(self):
        return "ListEntry(%r, %r, %r, %r, %r)" % (self.words, self.phonemes, self.text, self.flags, self.condition)


def parse_word_list(text, source="word list"):
    """Returns the entries of a word list in file order (the reader searches them from the last to the first)."""
    entries = []
    text_mode = False
    for number, raw in enumerate(text.split("\n"), 1):
        line = strip_comment(raw).strip()
        if not line:
            continue
        if line == "$textmode":
            text_mode = True
            continue
        if line == "$phonememode":
            text_mode = False
            continue
        condition = None
        m = re.match(r"\?(!?)(\d+)\s*", line)
        if m:
            condition = (int(m.group(2)), not m.group(1))
            line = line[m.end():]
            if not line:
                _fail(source, number, "a condition with no entry after it")
        if line[0] == "(" and ")" in line[2:]:
            close = line.index(")", 2)
            words = tuple(line[1:close].split())
            rest = line[close + 1:].split()
            if not words:
                _fail(source, number, "an empty word group")
        else:
            fields = line.split()
            words, rest = (fields[0],), fields[1:]
        first = [f for f in rest if not f.startswith("$")]
        flags = tuple(f[1:] for f in rest if f.startswith("$"))
        if any(not f for f in flags):
            _fail(source, number, "a `$` with no flag name")
        if text_mode or "text" in flags:
            entries.append(ListEntry(words, None, " ".join(first) if first else None, flags, condition, number))
        else:
            # A line with more fields than the format names is kept with its extra fields: the reader refuses it by
            # name if a text ever reaches it, and never reads a guess of what the line meant.
            entries.append(ListEntry(words, first[0] if first else None, None, flags, condition, number,
                                     tuple(first[1:])))
    return entries


# ---------------------------------------------------------------------------------------------------------------
# The voice file: one `<attribute> <values>` per line.

def parse_voice(text, source="voice file"):
    """Returns {attribute: [values of each line that sets it]}; a value is the list of the line's fields."""
    out = {}
    for number, raw in enumerate(text.split("\n"), 1):
        line = strip_comment(raw).strip()
        if not line:
            continue
        fields = line.split()
        out.setdefault(fields[0], []).append(fields[1:])
    return out


# ---------------------------------------------------------------------------------------------------------------
# The phoneme definitions (`phonemes` and the files it includes).

#: the master phoneme file of the phoneme folder, which includes the others
MASTER_FILE = "phonemes"
ONE_VALUE = ("starttype", "endtype", "length", "lengthmod", "ipa", "stress_type", "voicingswitch",
             "import_phoneme", "CALL")
REST_OF_LINE = ("Vowelin", "Vowelout")
CONDITION = re.compile(r"^(\w+)(?:\((.*)\))?$")
INSTRUCTION = re.compile(r"^(\w+)\((.*)\)$")


class PhonemeDef(object):
    __slots__ = ("name", "table", "words", "values", "program", "line")

    def __init__(self, name, table, line):
        self.name = name
        self.table = table
        self.words = []     # the bare words of the definition (type, features, properties), in order
        self.values = {}    # the one-value statements met outside any condition (ipa, starttype, stress_type ...)
        self.program = []   # the instructions: ("do", name, argument) and ("if", [(condition, body) ...], else body)
        self.line = line


class PhonemeSource(object):
    """What the master phoneme file says: tables (name -> parent, and their own definitions) and procedures."""

    def __init__(self):
        self.parents = {}       # table name -> parent table name, or None
        self.own = {}           # table name -> {phoneme name: PhonemeDef}, in definition order
        self.procedures = {}    # (table name, procedure name) -> PhonemeDef

    def table(self, name):
        """The definitions a table sees: its own over its parents', {phoneme name: PhonemeDef}."""
        if name not in self.parents:
            raise DataFileError("the phoneme source has no table %r; it has: %s" % (name, " ".join(self.parents)))
        chain = []
        while name is not None:
            chain.append(name)
            name = self.parents[name]
        seen = {}
        for t in reversed(chain):
            seen.update(self.own[t])
        return seen


def decode_ipa(value):
    """An `ipa` value: its characters, with `U+hhhh` read as the character of that code."""
    return re.sub(r"U\+([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), value)


def _unescape(name):
    return re.sub(r"\\(.)", r"\1", name)


def _tokens(text, read_include, source):
    """Yields (source, line number, fields) with comments cut and `include` files read in place."""
    for number, raw in enumerate(text.split("\n"), 1):
        fields = strip_comment(raw).split()
        if not fields:
            continue
        if fields[0] == "include":
            if len(fields) != 2:
                _fail(source, number, "`include` takes one file name")
            for item in _tokens(read_include(fields[1]), read_include, fields[1]):
                yield item
        else:
            yield source, number, fields


def _condition(fields, source, number):
    """`[NOT] test(attribute) {AND|OR [NOT] test(attribute)}` -> (joiner, [(negated, test, attribute) ...])."""
    tests, joiner, negated = [], None, False
    for f in fields:
        if f in ("AND", "OR"):
            if joiner not in (None, f):
                _fail(source, number, "a condition mixes AND and OR")
            joiner = f
        elif f == "NOT":
            negated = True
        else:
            m = CONDITION.match(f)
            if not m:
                _fail(source, number, "a condition test that is not `name(attribute)`: %s" % f)
            tests.append((negated, m.group(1), m.group(2) or ""))
            negated = False
    if not tests:
        _fail(source, number, "an empty condition")
    return (joiner or "AND", tests)


def _wanted_tables(text, tables):
    """The named tables and their ancestors, from the master file's own `phonemetable` lines."""
    parents = {"base": None}
    for raw in text.split("\n"):
        fields = strip_comment(raw).split()
        if len(fields) == 3 and fields[0] == "phonemetable":
            parents[fields[1]] = None if fields[2] == "_" else fields[2]
    wanted = set()
    for name in tables:
        while name is not None:
            if name not in parents:
                raise DataFileError("the phoneme source has no table %r" % name)
            wanted.add(name)
            name = parents[name]
    return wanted


def parse_phoneme_source(text, read_include, source="phonemes", tables=None):
    """Parses the master phoneme file. `read_include(name)` returns the text of an included file. With `tables`,
    only those tables and their ancestors are read: the files the other tables include are not asked for."""
    if tables is not None:
        wanted = _wanted_tables(text, tables)
        kept, keep = [], True
        for raw in text.split("\n"):
            fields = strip_comment(raw).split()
            if len(fields) == 3 and fields[0] == "phonemetable":
                keep = fields[1] in wanted
            kept.append(raw if keep else "")
        text = "\n".join(kept)
    builder = _Builder()
    for src, number, fields in _tokens(text, read_include, source):
        if builder.current is None:
            builder.outside(src, number, fields)
        else:
            builder.inside(src, number, fields)
    if builder.current is not None:
        raise DataFileError("%s: the definition of %r is not ended" % (source, builder.current.name))
    return builder.out


class _Builder(object):
    """Receives the lines of the phoneme files one by one and fills a PhonemeSource."""

    def __init__(self):
        self.out = PhonemeSource()
        self.table = "base"
        self.out.parents[self.table] = None
        self.out.own[self.table] = {}
        self.current = None     # the PhonemeDef being read
        self.stack = None       # the open bodies: stack[-1] receives the next instruction
        self.open_ifs = None    # the open ("if", branches, else) nodes

    def outside(self, src, number, fields):
        """A line between two definitions: a new table, or the start of a phoneme or a procedure."""
        head, out = fields[0], self.out
        if head == "phonemetable":
            if len(fields) != 3:
                _fail(src, number, "`phonemetable` takes a name and a parent")
            parent = None if fields[2] == "_" else fields[2]
            if parent is not None and parent not in out.parents:
                _fail(src, number, "the parent table %r is not defined before %r" % (parent, fields[1]))
            self.table = fields[1]
            out.parents[self.table] = parent
            out.own[self.table] = {}
        elif head in ("phoneme", "procedure"):
            if len(fields) < 2:
                _fail(src, number, "`%s` takes a name" % head)
            self.current = PhonemeDef(_unescape(fields[1]), self.table, number)
            self.stack, self.open_ifs = [self.current.program], []
            if head == "phoneme":
                out.own[self.table][self.current.name] = self.current
            else:
                out.procedures[(self.table, self.current.name)] = self.current
            self.current.words.extend(fields[2:])    # `phoneme #@ virtual`: the type may follow the name
        else:
            _fail(src, number, "%r outside a phoneme definition" % head)

    def inside(self, src, number, fields):
        """A line of a definition: its end, or statements."""
        head = fields[0]
        if head in ("endphoneme", "endprocedure"):
            if self.open_ifs:
                _fail(src, number, "the definition of %r ends inside an IF" % self.current.name)
            self.current = None
            return
        if head in ("phoneme", "procedure", "phonemetable"):
            _fail(src, number, "the definition of %r is not ended" % self.current.name)
        i = 0
        while i < len(fields):
            i = self.statement(src, number, fields, i)

    def branch(self, src, number, fields, i):
        """IF / ELIF / ELSE / ENDIF at fields[i]; returns the index after it."""
        f = fields[i]
        if f != "IF" and not self.open_ifs:
            _fail(src, number, "%s without IF" % f)
        if f == "ELSE":
            self.stack[-1] = self.open_ifs[-1][2]
            return i + 1
        if f == "ENDIF":
            self.open_ifs.pop()
            self.stack.pop()
            return i + 1
        if "THEN" not in fields[i:]:
            _fail(src, number, "%s without THEN on its line" % f)
        then = fields.index("THEN", i)
        cond = _condition(fields[i + 1:then], src, number)
        if f == "IF":
            node = ("if", [], [])
            self.stack[-1].append(node)
            self.open_ifs.append(node)
            self.stack.append(None)
        body = []
        self.open_ifs[-1][1].append((cond, body))
        self.stack[-1] = body
        return then + 1

    def statement(self, src, number, fields, i):
        """Reads the statement at fields[i]; returns the index of the next one."""
        f, body = fields[i], self.stack[-1]
        if f in ("IF", "ELIF", "ELSEIF", "ELSE", "ENDIF"):
            return self.branch(src, number, fields, i)
        if f in REST_OF_LINE:
            return len(fields)
        if f in ONE_VALUE:
            if i + 1 >= len(fields):
                _fail(src, number, "`%s` takes one value" % f)
            value = decode_ipa(fields[i + 1]) if f == "ipa" else fields[i + 1]
            if f in ("CALL", "import_phoneme") or (f == "ipa" and self.open_ifs):
                body.append(("do", f, value))
            if not self.open_ifs:
                self.current.values[f] = value
            return i + 2
        j = i
        if "(" in f and ")" not in f:       # an instruction whose arguments hold spaces
            while j < len(fields) - 1 and ")" not in fields[j]:
                j += 1
            f = " ".join(fields[i:j + 1])
        m = INSTRUCTION.match(f)
        if m:
            body.append(("do", m.group(1), m.group(2)))
        elif "(" in f:
            _fail(src, number, "an instruction that does not close its bracket: %s" % f)
        elif f == "RETURN":
            body.append(("do", "RETURN", ""))
        elif not self.open_ifs:
            self.current.words.append(f)
        return j + 1


def read_phoneme_source(folder, tables=None):
    """Reads the master phoneme file of a folder (MASTER_FILE) and the files it includes (relative to that folder)."""
    def read(name):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise DataFileError("the phoneme file %s is absent: fetch the pinned data of the processor" % path)
        with open(path, encoding="utf-8") as f:
            return f.read()
    return parse_phoneme_source(read(MASTER_FILE), read, MASTER_FILE, tables)
