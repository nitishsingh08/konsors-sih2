"""
Static checks on the Colab driver cells.

These are not unit tests of behaviour -- the cells need Drive to run. They
check the mistakes that are invisible until a multi-million-row run dies on
line 200 of a cell: a name that is local when it was meant to be global, an
undefined reference, a bare `raise SystemExit` that IPython renders as a crash,
or a module-level banner that makes a no-op look like a running job.

Run:  python3 -m unittest test_colab_cells
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent

#: Cells that import the repo modules and touch Drive.
DRIVER_CELLS = (
    "colab_p3a_industrial.py",
    "colab_p4a_tier1.py",
    "colab_p4a_lineage.py",
    "colab_p1b_external_inventory.py",
    "colab_recon.py",
    "colab_survey_tree.py",
    "colab_p0_gap.py",
    "colab_p7_train.py",
    "colab_p6_dataset.py",
)

#: The bootstrap and diagnostics are exempt: they have no driver body.
LIGHTWEIGHT = (
    "colab_bootstrap.py",
    "colab_p3a_diagnose_external.py",
    "colab_p3a_remount_drive.py",
)


def _loads_outside_nested_scopes(node):
    """Names READ inside `node`, not descending into nested def/class bodies.

    Those have their own scope and parameters; reporting their names against the
    enclosing scope produces only false positives.
    """
    out = []

    def walk(n):
        for sub in ast.iter_child_nodes(n):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef, ast.Lambda)):
                continue                      # own scope, checked separately
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                out.append(sub)
            walk(sub)

    walk(node)
    return out


def parse(name):
    return ast.parse((REPO / name).read_text())


def module_level_names(tree):
    """Names bound at module scope, including tuple-unpacked and if-branch ones."""
    names = set()

    def bind(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for element in target.elts:
                bind(element)

    def walk(body):
        # Imports bind wherever they appear, including inside try/except or if
        # blocks -- `try: import lightgbm as lgb` is the common case.
        for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    bind(target)
            elif isinstance(node, ast.AnnAssign):
                bind(node.target)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.If):
                walk(node.body)
                walk(node.orelse)

    walk(tree.body)
    return names


def function_scopes(tree):
    """Functions DEFINED at module level.

    Nested functions are deliberately excluded: they must be resolved against
    their enclosing scope, and `ast.walk` would also surface them as
    independent entry points, reporting legitimate closure references as
    undefined.
    """
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


class TestCellsParse(unittest.TestCase):
    def test_every_cell_parses(self):
        for name in DRIVER_CELLS + LIGHTWEIGHT:
            with self.subTest(cell=name):
                self.assertIsInstance(parse(name), ast.Module)

    def test_driver_cells_declare_a_guard(self):
        """Each must gate its compute body so inspect mode cannot run it."""
        for name in ("colab_p3a_industrial.py", "colab_p4a_tier1.py"):
            with self.subTest(cell=name):
                tree = parse(name)
                guards = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "not _STOP"
                ]
                self.assertEqual(len(guards), 1, f"{name} lacks the _STOP guard")
                called = ast.unparse(guards[0].body[0])
                self.assertIn("run_full_pass()", called)


def assigned_in(node):
    """Names bound by assignment/import/for/with/comprehension inside `node`."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            names.add(sub.id)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
    return names


class TestNoShadowedModuleConstants(unittest.TestCase):
    """A module-level constant ASSIGNED inside a function becomes local.

    This is exactly what broke the tier-1 pass: wrapping the body in a function
    turned `DATE_BATCH = max(1, int(DATE_BATCH))` into an UnboundLocalError on
    the first line of a long run. Only the shadowing case is dangerous, so only
    that case is flagged -- a genuine local such as `geoms = ...geoms...` is
    normal and must not trip the check.
    """

    def test_no_module_constant_is_reassigned_inside_a_function(self):
        offenders = []
        for name in DRIVER_CELLS + LIGHTWEIGHT:
            tree = parse(name)
            module_names = module_level_names(tree)
            for fn in function_scopes(tree):
                # Names this function binds locally; a module constant that
                # appears here is being shadowed.
                local = {a.arg for a in fn.args.args}
                local |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id in module_names
                                and target.id not in assigned_in(fn)
                            ):
                                offenders.append(
                                    f"{name}:{fn.name}() line {sub.lineno}: "
                                    f"shadows module constant {target.id}"
                                )
        self.assertEqual(offenders, [], "\n".join(offenders))


class TestNoUndeclaredAccumulators(unittest.TestCase):
    """`weak_counts[cls] += n` without a declaration survived review and only
    surfaced at partition 0 of a 7.27M-event pass. Any name augmented inside a
    function must be assigned in that same function."""

    def test_no_augmented_name_is_undeclared(self):
        offenders = []
        for name in DRIVER_CELLS + LIGHTWEIGHT:
            tree = parse(name)
            for fn in function_scopes(tree):
                bound = set()
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        bound.add(sub.id)
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for alias in sub.names:
                            bound.add(
                                (alias.asname or alias.name).split(".")[0]
                            )
                # Catch EVERY way an accumulator is mutated, not just `+=`:
                #   NAME += ...            (AugAssign)
                #   NAME[k] += ...         (AugAssign on a Subscript)
                #   NAME[k] = ...          (Assign on a Subscript)
                #   NAME.setdefault(...)   (a Call on Name)
                # `missing_pairs` was used only via setdefault, which no
                # AugAssign check could see, and it survived a full P6 run.
                for sub in ast.walk(fn):
                    target = None
                    if isinstance(sub, ast.AugAssign):
                        target = sub.target
                    elif isinstance(sub, ast.Assign):
                        target = sub.targets[0]
                    if target is None:
                        continue
                    root = target
                    while isinstance(root, (ast.Subscript, ast.Attribute)):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id not in bound:
                        offenders.append(
                            f"{name}:{fn.name}() line {sub.lineno}: "
                            f"'{root.id}' mutated but never assigned"
                        )
                    elif isinstance(root, ast.Name):
                        continue
                for sub in ast.walk(fn):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and isinstance(sub.func.value, ast.Name)
                            and sub.func.attr
                            in {"setdefault", "append", "add", "update",
                                "extend", "insert"}
                            and sub.func.value.id not in bound):
                        offenders.append(
                            f"{name}:{fn.name}() line {sub.lineno}: "
                            f"'{sub.func.value.id}.{sub.func.attr}()' "
                            "called but never assigned"
                        )
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_p6_declares_the_accumulators_it_updates(self):
        """colab_p6_dataset.py accumulates five counters across partitions; a
        missing declaration only shows up at partition 0 of the full pass."""
        source = (REPO / "colab_p6_dataset.py").read_text()
        for counter in ("label_counts", "rule_counts", "weak_counts"):
            self.assertIn(f"{counter} = Counter()", source, counter)
        self.assertIn("null_counts = {c: 0 for c in ALL_FEATURES}", source)
        # missing_pairs is a dict of sets, not a Counter: counting distinct
        # events per input is the whole point, and a Counter summed per-string
        # counts double-counted.
        self.assertIn("missing_pairs: dict = {}", source)
        self.assertNotIn("missing_counter = Counter()", source)


class TestNoUseBeforeAssignment(unittest.TestCase):
    """Straight-line cell scripts read top to bottom, so a name used above its
    assignment is simply a NameError waiting to happen.

    `train_idx` was referenced in the leakage assertion 46 lines above the line
    that assigns it, after an over-eager edit replaced the split masks with the
    sample indices. Both an undefined-name check and a mutation check pass that
    cleanly -- the name exists, just too late.
    """

    def _scan(self, body, available, path, where, offenders):
        """Linear walk of one statement list, tracking names bound so far.

        Two things it must NOT do:

        * descend into a nested function or class body -- those have their own
          scope and their own parameters, checked separately;
        * report a name that is bound later in the SAME statement (a `for` target
          read by its own iterable is a bug only if the iterable is the name).
        """
        bound = set(available)
        for node in body:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    bound.add(sub.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    bound.add(sub.name)
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    bound.add(sub.name)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                # A def binds only its own name here; its body is a separate
                # scope and is scanned on its own pass.
                continue
            for sub in _loads_outside_nested_scopes(node):
                if sub.id not in bound:
                    offenders.append(
                        f"{path}:{where} line {sub.lineno}: "
                        f"'{sub.id}' used before assignment"
                    )

    @staticmethod
    def _hoisted(tree):
        """Names available before ANY statement runs.

        Python hoists imports and def/class bindings, so those are genuinely
        pre-bound. Ordinary assignments are NOT: seeding with them would hide
        the exact bug this check exists for, since a name assigned 46 lines
        later would look available at line 150.
        """
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                names.add(node.name)
        return names

    def test_no_use_before_assignment(self):
        import builtins

        offenders = []
        for name in DRIVER_CELLS + LIGHTWEIGHT:
            tree = parse(name)
            available = self._hoisted(tree) | set(dir(builtins))
            self._scan(tree.body, available, name, "module", offenders)
            # A function body executes AFTER the module body finishes, so it
            # may read any module global -- including one assigned later at
            # module level. Only its own parameters start unbound. Seeding with
            # `available` (hoists only) here would report every global read as
            # a use-before-assignment.
            globals_visible = module_level_names(tree) | set(dir(builtins))
            for fn in [n for n in tree.body
                       if isinstance(n, ast.FunctionDef)]:
                local = {a.arg for a in fn.args.args}
                local |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
                self._scan(fn.body, globals_visible | local, name,
                           f"{fn.name}()", offenders)
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_check_detects_a_planted_regression(self):
        """Prove the check would have caught the bug it was written for."""
        planted = ast.parse(
            "def f():\n"
            "    use(a, b)\n"
            "    a = 1\n"
            "    b = 2\n"
        )
        offenders = []
        available = module_level_names(planted) | set(dir(__builtins__))
        fn = planted.body[0]
        local = {x.arg for x in fn.args.args}
        self._scan(fn.body, available | local, "planted", "f()", offenders)
        self.assertTrue(any("used before assignment" in o for o in offenders),
                        offenders)


class TestNoUndefinedNames(unittest.TestCase):
    """Every free name inside a function must resolve at module scope.

    Cross-function references and function-local imports are resolved too: a
    function defined in this module is module-scope, and an import statement
    inside a function binds locally.
    """

    def test_no_undefined_free_names(self):
        """Resolve each function with its enclosing scopes available.

        A nested function closes over the enclosing scope, so `to_target`
        assigned in `run_full_pass` is legitimately visible inside
        `project_points` defined within it. Walking each function in isolation
        would call that undefined, which is why the closure chain is followed.
        """
        import builtins

        builtins_set = set(dir(builtins))
        offenders = []

        def scope_names(fn, outer=()):
            names = set(outer)
            names |= {a.arg for a in fn.args.args}
            # Lambda parameters bind inside the enclosing body, so they must be
            # visible to the loads inside that lambda.
            for sub in ast.walk(ast.Module(body=list(fn.body), type_ignores=[])):
                if isinstance(sub, ast.Lambda):
                    names |= {a.arg for a in sub.args.args}
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    names.add(sub.name)
            names |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
            if fn.args.vararg:
                names.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                names.add(fn.args.kwarg.arg)
            names |= assigned_in(fn)
            # Nested defs bind their own names here but their bodies are
            # checked separately, with this scope as their closure.
            return names

        def body_nodes(fn):
            """Yield nodes in `fn`'s OWN body, never descending into a nested
            def/class. ast.walk would also yield the nested body, so its loads
            get checked against the outer scope and every closure variable
            looks undefined."""
            def walk(n):
                for sub in ast.iter_child_nodes(n):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef)):
                        continue      # own scope, handled separately
                    if isinstance(sub, ast.Lambda):
                        yield sub    # lambda, but its ARGUMENTS are in scope
                    yield sub
                    walk(sub)

            stack = list(fn.body)
            while stack:
                node = stack.pop()
                yield node
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    continue
                stack.extend(walk(node))

        def check(fn, outer, label, seen):
            if id(fn) in seen:
                return
            seen = seen | {id(fn)}
            available = scope_names(fn, outer) | builtins_set
            for node in body_nodes(fn):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check(node, available, label, seen)
                elif (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id not in available
                ):
                    offenders.append(f"{label} line {node.lineno}: {node.id}")

        for name in DRIVER_CELLS + LIGHTWEIGHT:
            tree = parse(name)
            module = module_level_names(tree) | builtins_set
            for fn in function_scopes(tree):
                check(fn, module, f"{name}:{fn.name}()", frozenset())
        self.assertEqual(sorted(set(offenders)), [], "\n".join(sorted(set(offenders))))


class TestNoCrashLookingStops(unittest.TestCase):
    """`raise SystemExit(0)` is a deliberate stop but IPython prints a traceback
    that reads like a crash. Cells must use the flag guard instead."""

    def test_no_zero_exit_system_exit(self):
        """`SystemExit(0)` is a *deliberate* stop that IPython renders as a
        traceback reading like a crash. `SystemExit(1)` is a genuine failure and
        is exactly what should stop the cell, so only the zero form is banned."""
        for name in DRIVER_CELLS + LIGHTWEIGHT:
            with self.subTest(cell=name):
                source = (REPO / name).read_text()
                self.assertNotIn("SystemExit(0)", source)
                self.assertNotIn("SystemExit(0)", source.replace(" ", ""))

    def test_inspect_mode_is_announced(self):
        """A stale pasted cell must be obvious from the first lines."""
        for name in ("colab_p3a_industrial.py", "colab_p4a_tier1.py"):
            with self.subTest(cell=name):
                source = (REPO / name).read_text()
                self.assertIn("MODE:", source)
                self.assertIn("INSPECT_ONLY", source)

    def test_inspect_branch_states_that_nothing_was_written(self):
        for name in ("colab_p3a_industrial.py", "colab_p4a_tier1.py"):
            with self.subTest(cell=name):
                source = (REPO / name).read_text()
                self.assertIn("Nothing was written", source)


class TestBannerPlacement(unittest.TestCase):
    """A banner printed outside the guard makes a no-op look like a running job."""

    def test_full_pass_banner_is_inside_the_guarded_function(self):
        for name in ("colab_p3a_industrial.py", "colab_p4a_tier1.py"):
            with self.subTest(cell=name):
                tree = parse(name)
                fn = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_full_pass"
                )
                inside = "\n".join(
                    (REPO / name).read_text().split("\n")[fn.lineno - 1:fn.end_lineno]
                )
                # P3a labels its body "STEP 3/4"; tier 1 uses an explicit
                # "FULL PASS" banner. Either way the banner must be guarded.
                self.assertRegex(inside, r"FULL PASS|P4a TIER 1|P3a STEP 3")
                # And no module-level statement may print it.
                for node in tree.body:
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        rendered = ast.unparse(node)
                        if "FULL PASS" in rendered and "MODE" not in rendered:
                            self.fail(f"{name}: module-level FULL PASS banner")

    def test_tier1_banner_mentions_that_it_writes(self):
        source = (REPO / "colab_p4a_tier1.py").read_text()
        self.assertRegex(source, r"FULL PASS.*\(writing")

    def test_p3a_body_banner_is_inside_run_full_pass(self):
        tree = parse("colab_p3a_industrial.py")
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_full_pass"
        )
        inside = "\n".join(
            (REPO / "colab_p3a_industrial.py")
            .read_text()
            .split("\n")[fn.lineno - 1:fn.end_lineno]
        )
        self.assertIn("P3a STEP 3", inside)


class TestHardcodedDrivePaths(unittest.TestCase):
    """Paths must go through the resolver, not a literal.

    A literal `/content/drive/MyDrive/Firm_Clusters/...` broke twice: once when
    the tree moved to a duplicated folder, and once when a stale mount hid it.
    """

    def test_no_bare_mydrive_literals(self):
        pattern = "/content/drive/MyDrive/Firm_Clusters"
        offenders = []
        for name in DRIVER_CELLS:
            with self.subTest(cell=name):
                for i, line in enumerate(
                    (REPO / name).read_text().split("\n"), start=1
                ):
                    if pattern in line and not line.lstrip().startswith("#"):
                        offenders.append(f"{name}:{i}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_cells_use_the_resolver(self):
        for name in DRIVER_CELLS:
            with self.subTest(cell=name):
                source = (REPO / name).read_text()
                self.assertIn("resolve_project_dir", source)


if __name__ == "__main__":
    unittest.main()