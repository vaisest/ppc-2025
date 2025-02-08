"""
Document Builder utilities to enable unified output generation for both terminal and web front-end

To achieve this, we generate output by building up a very simple DOM model, consisting of (styled) strings,
paragraphs, and lists. We don't allow nesting, i.e., each paragraph is a sequence of styled strings, each list
is a sequence of paragraphs, and a document is a sequence of paragraphs and lists.

In this file, we first define dataclasses for the output structure itself, and then some builder utilities
as a convenient means of building up a document. Finally, we provide  a set of functions for rendering the
result to html or terminal.
"""

from typing import List, Union, TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager
from ppcgrader import quantity
import textwrap
if TYPE_CHECKING:
    from quantity import Quantity
    # markup is only needed for web output generation, and to specify type hints.
    # we *cannot* just import it unconditionally, because we don't want to require
    # that students have this package installed.
    from markupsafe import Markup

##############################################################################
#      Node definitions
##############################################################################


class NodeData:
    """
    Base class for all elements in the document.
    """
    pass


@dataclass
class StringNode(NodeData):
    """
    A `StringNode` is similar to an HTML span, representing a text and (potentially) an associated style.

    For HTML, this node could correspond to a true `<span>`, but also to `<em>` or `<strong>` elements.
    For terminal output, we map styles to ANSI codes.

    Multiple `StringNode`s can be concatenated together to a `TextNode`.
    """
    content: str
    style: str

    # Define addition operators, so we can concatenate these like actual strings
    def __add__(self, other: Union['StringNode', str]):
        if isinstance(other, StringNode):
            return TextNode(content=[self, other])
        else:
            return TextNode(content=[self, StringNode(other, '')])

    def __radd__(self, other: Union['StringNode', str]):
        if isinstance(other, StringNode):
            return TextNode(content=[other, self])
        else:
            return TextNode(content=[StringNode(other, ''), self])


@dataclass
class TextNode(NodeData):
    """
    A `TextNode` is similar to an HTML `<p>`, representing the concatenation of multiple pieces of text in potentially
    different styles.
    """
    content: List[StringNode]

    # Define addition operators, so we can concatenate these like actual strings
    def __add__(self, other: Union['StringNode', 'TextNode', str]):
        if isinstance(other, StringNode):
            return TextNode(content=self.content + [other])
        elif isinstance(other, TextNode):
            return TextNode(content=self.content + other.content)
        else:
            return TextNode(content=self.content +
                            [StringNode(other, style='')])

    def __radd__(self, other: Union['StringNode', str]):
        if isinstance(other, StringNode):
            return TextNode(content=[other] + self.content)
        elif isinstance(other, TextNode):
            return TextNode(content=other.content + self.content)
        else:
            return TextNode(content=[StringNode(other, '')] + self.content)


@dataclass
class ListNode(NodeData):
    """
    A `ItemList` node represents a (unordered) list of `TextNode` objects.
    """
    items: List[TextNode]


@dataclass
class Document(NodeData):
    """
    A `Document` is a sequence of `TextNode' and `ListNode` objects.
    """
    content: List[Union[TextNode, ListNode]]


##############################################################################
#      Builder utility
##############################################################################


class Builder:
    """
    Base class for builder objects.

    Builders are expected to be used in (hierarchical) with statements, that reflect the
    structure of the document being built. The `group` function defines a context-manager
    that suppresses `QtyNotSetError`; this allows to group output steps together, and if
    one of them fails because of a missing quantity, the rest of the output still gets
    generated.

    While we try to unify web and terminal output, there are some ways in which they will
    be different. In particular, we often want the terminal to say the same thing as web,
    but with terser wording, so it works better with an 80-character line limit. Thus, the
    builder object is expected to specify whether it works in "web" or "term" `mode`, and
    the `alt` function allows to select between two corresponding text options.
    """
    def build(self) -> NodeData:
        raise NotImplementedError()

    @property
    def mode(self) -> str:
        raise NotImplementedError()

    def alt(self, *, web: str, term: str) -> str:
        """
        Returns the `web` argument in web mode, and the `term` argument in terminal mode.
        """
        if self.mode == "web":
            return web
        elif self.mode == "term":
            return term

    @contextmanager
    def group(self):
        try:
            yield
        except quantity.QtyNotSetError:
            pass


class TextBuilder(Builder):
    """
    Utility for building up a `TextNode` by concatenating strings, `StringNode`s, and other `TextNode`s.
    """
    def __init__(self, parent: Builder, content=None):
        self.parent = parent
        self.content = [] if content is None else content

    def __iadd__(self, other: Union[str, StringNode, TextNode]):
        if isinstance(other, str):
            self.content.append(StringNode(other, ''))
        elif isinstance(other, TextNode):
            self.content += other.content
        elif isinstance(other, StringNode):
            self.content.append(other)
        else:
            assert False
        return self

    def build(self) -> TextNode:
        return TextNode(content=self.content)

    @property
    def mode(self) -> str:
        return self.parent.mode


class ListBuilder(Builder):
    """
    Utility for building up a `ListNode`.

    Provides a context-manager function `item` for generating list items, and
    `add_item` for directly adding them to the builder.
    """
    def __init__(self, parent: Builder):
        self.items = []
        self.parent = parent

    @contextmanager
    def item(self):
        with self.group():
            builder = TextBuilder(parent=self)
            yield builder
            self.items.append(builder.build())

    def add_item(self, content: Union[str, TextNode]):
        if isinstance(content, str):
            self.items.append(TextNode([StringNode(content=content,
                                                   style="")]))
        elif isinstance(content, TextNode):
            self.items.append(content)
        else:
            pass
            #assert False

    def build(self) -> ListNode:
        return ListNode(self.items)

    @property
    def mode(self) -> str:
        return self.parent.mode


class DocumentBuilder(Builder):
    """
    Utility for building up a `Document`.

    Provides context managers for creating `text` and `list` sub-builders.
    """
    def __init__(self, mode: str):
        self.content = []
        self._mode = mode

    @contextmanager
    def list(self):
        builder = ListBuilder(self)
        yield builder
        self.content.append(builder.build())

    @contextmanager
    def text(self):
        builder = TextBuilder(self)
        yield builder
        self.content.append(builder.build())

    def build(self) -> Document:
        return Document(self.content)

    @property
    def mode(self) -> str:
        return self._mode


def em(text: str):
    """
    Create a `StringNode` with "em" style.
    """
    return StringNode(str(text), style="em")


def strong(text: "Union[str, Quantity]"):
    """
    Create a `StringNode` with "strong" style.
    """
    return StringNode(str(text), style="strong")


##############################################################################
#      Rendering definitions
##############################################################################


class TerminalPrinter:
    """
    TerminalPrinter specifies how text styles are represented on the terminal,
    and how lists are formatted.

    It has the following attributes:
    format: mapping from format strings to prefix and suffix that surround such text.
            `color` mode, these will be ansi codes.
    list_item: String that introduces a new list item
    list_sep: String that appears at the end of each list item.
    list_indent: String that is used to indent the entire list.
    """
    def __init__(self, color: bool):
        self.color = color
        self.format = {"strong": ("", ""), "em": ("", "")}

        if color:
            self.format["strong"] = ('\033[34m', '\033[0m')
            self.format["em"] = ('\033[1m', '\033[0m')

        self.list_item = " · "
        self.list_indent = ""
        self.list_sep = "\n"

    def format_string(self, text: str, style: str) -> str:
        """
        Format `text` according to `style` as specified in self.format.
        """
        beg, end = self.format.get(style, ("", ""))
        return f"{beg}{text}{end}"

    def format_list_item(self, item: str) -> str:
        """
        Format a single list item of the given text.
        """
        text = f"{self.list_item}{item}"
        lil = len(self.list_item)
        text = textwrap.indent(text, ' ' * lil)
        text = text[lil:]
        return f"{text}\n{self.list_sep}"


def _generate_node_term(node: NodeData, printer: TerminalPrinter) -> str:
    """
    Generate a terminal string representation for the given node,
    recursively stringifying sub-nodes.
    """
    if isinstance(node, StringNode):
        return printer.format_string(node.content, node.style)
    elif isinstance(node, TextNode):
        result = ""
        for s in node.content:
            result += _generate_node_term(s, printer)
        return result
    elif isinstance(node, ListNode):
        result = ""
        for item in node.items:
            item_text = _generate_node_term(item, printer)
            result += printer.format_list_item(item_text.strip())
        return textwrap.indent(result, printer.list_indent)
    elif isinstance(node, Document):
        result = ""
        for item in node.content:
            item_text = _generate_node_term(item, printer)
            result += f"{item_text}\n"
        return result.rstrip()


def generate_term(doc: Document, printer: TerminalPrinter) -> str:
    """
    Generate terminal string for the document.
    """
    return _generate_node_term(doc, printer)


def _generate_node_html(node: NodeData) -> "Markup":
    """
    Generate a web string representation for the given node,
    recursively stringifying sub-nodes.
    """
    from markupsafe import escape, Markup

    if isinstance(node, StringNode):
        safe = escape(node.content)
        if node.style == "":
            return safe
        else:
            return Markup(f"<{node.style}>{safe}</{node.style}>")
    elif isinstance(node, TextNode):
        result = Markup("")
        for s in node.content:
            result += _generate_node_html(s)
        return result
    elif isinstance(node, ListNode):
        result = ""
        for item in node.items:
            item_text = _generate_node_html(item)
            result += Markup("<li>") + item_text + Markup("</li>\n")
        return Markup("<ul>") + result + Markup("</ul>")
    elif isinstance(node, Document):
        result = Markup("")
        for item in node.content:
            item_text = _generate_node_html(item)
            result += item_text + "\n"
        if result.endswith("\n\n"):
            return result[:-1]
        return result


def generate_html(doc: Document):
    """
    Generate web string for the document.
    """
    return _generate_node_html(doc)
