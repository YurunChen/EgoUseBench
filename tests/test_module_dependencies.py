"""Keep presentation and command adapters outside the domain implementation."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [("egocua", {"tools"}), ("sites", {"egocua", "tools"})],
)
def test_domain_imports_do_not_depend_on_adapters(package, forbidden):
    violations = []
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)


def test_core_modules_have_no_import_cycles():
    modules = {
        ".".join(path.relative_to(ROOT).with_suffix("").parts): path
        for package in ("egocua", "sites")
        for path in (ROOT / package).rglob("*.py")
    }
    dependencies = {name: set() for name in modules}
    for name, path in modules.items():
        package = name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                targets = [target, *(f"{target}.{alias.name}" for alias in node.names)]
            else:
                continue
            dependencies[name].update(target for target in targets if target in modules)

    visited = set()
    active = []

    def visit(name):
        assert name not in active, " -> ".join([*active, name])
        if name in visited:
            return
        active.append(name)
        for target in sorted(dependencies[name]):
            visit(target)
        active.pop()
        visited.add(name)

    for name in sorted(modules):
        visit(name)
