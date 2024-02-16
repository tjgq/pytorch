# Owner(s): ["module: dynamo"]
import io
import logging
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch._dynamo.utils import counters


class ReorderLogsTests(torch._dynamo.test_case.TestCase):
    def test_dont_reorder_print(self):
        def f(x):
            x = x + x
            print("moo")
            x = x * x
            return x

        counters.clear()
        x = torch.randn(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertEqual(printed_output, "moo")
        self.assertEqual(len(counters["graph_break"]), 1)

    @torch._dynamo.config.patch(reorder_logs=True)
    def test_reorder_print(self):
        def f(x):
            print("moo")
            x1 = x + x
            print(x1)
            x2 = x1 * x1
            print(1, 2, 3)
            x3 = x2 + x2
            return (x1, x3)

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"moo\n{torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    @torch._dynamo.config.patch(reorder_logs=True)
    def test_reorder_warnings(self):
        import warnings

        def f(x):
            x1 = x + x
            warnings.warn("moo")
            x2 = x1 * x1
            warnings.warn(f"{x2}")
            x3 = x2 + x2
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with warnings.catch_warnings(record=True) as w:
            opt_out = opt_f(x)
            warning_messages = [str(i.message) for i in w]
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertIn("moo", warning_messages)

    @torch._dynamo.config.patch(reorder_logs=True)
    def test_reorder_print_graph_break(self):
        def f(x):
            x1 = x + x
            print(f"res: {x1}")
            x2 = x1 * x1
            torch._dynamo.graph_break()
            x3 = x2 + x2
            print(1, 2, 3)
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"res: {torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    @torch._dynamo.config.patch(reorder_logs=True)
    def test_reorder_custom_log_fn(self):
        custom_logs = []

        def custom_log(s: str):
            custom_logs.append(s)

        modified_logging_functions = {custom_log}
        modified_logging_functions.update(
            torch._dynamo.config.reorderable_logging_functions
        )

        def f(x):
            custom_log("moo")
            x1 = x + x
            custom_log(f"{x1}")
            return x + x

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with torch._dynamo.config.patch(
            reorderable_logging_functions=modified_logging_functions
        ):
            opt_out = opt_f(x)
        orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertEqual(custom_logs[0], "moo")
        self.assertEqual(custom_logs[1], f"{torch.ones(3, 3) * 2}")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
