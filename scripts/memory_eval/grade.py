"""Run inside the isolated grader process, with input passed over stdin."""

import ast
import builtins
import json
import resource
import sys

MODULES = {"codecs", "collections", "hashlib", "itertools", "math", "re", "typing"}
NAMES = """
    abs all any bool bytes bytearray dict enumerate filter frozenset int isinstance
    issubclass len list map max min object ord range reversed set sorted str sum tuple zip
    Exception IndexError KeyError NotImplementedError TypeError UnicodeDecodeError ValueError
""".split()


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError(name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def validate(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attribute access is outside the grading subset")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are outside the grading subset")
        if isinstance(node, ast.alias) and node.name.startswith("_"):
            raise ValueError("Private imports are outside the grading subset")
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError("Expected a standalone helper function")


def grade(code, checks):
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in NAMES}
                 | {"__import__": restricted_import}}
    try:
        tree = ast.parse(code)
        validate(tree)
        exec(compile(tree, "<answer>", "exec"), namespace)
    except Exception as exc:
        return [[label, False, type(exc).__name__] for label in checks]

    results = []
    for label, expression in checks.items():
        try:
            passed = bool(eval(expression, namespace))
            results.append([label, passed, "" if passed else "Wrong result"])
        except Exception as exc:
            results.append([label, False, type(exc).__name__])
    return results


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    payload = json.load(sys.stdin)
    print(json.dumps(grade(**payload)))
