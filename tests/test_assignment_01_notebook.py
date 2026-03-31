import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "assignment_01.ipynb"
ASSIGNMENT_RESULTS_PATH = ROOT / "assignment_01.xlsx"
TASK4_RESULTS_PATH = ROOT / "task4_prompt_experiment.xlsx"
TASK5_RESULTS_PATH = ROOT / "task5_judge_results.xlsx"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def cell_source(index: int) -> str:
    notebook = load_notebook()
    return "".join(notebook["cells"][index]["source"])


def exec_cells(indices, namespace=None):
    ns = {"__builtins__": __builtins__}
    if namespace:
        ns.update(namespace)
    for index in indices:
        exec(cell_source(index), ns)
    return ns


def summary_metrics():
    df = pd.read_excel(TASK5_RESULTS_PATH, sheet_name="summary")
    return dict(zip(df["metric"], df["value"]))


class FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    def __init__(self, content=None, parsed=None):
        self.content = content
        self.parsed = parsed


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, *, content=None, parsed=None, prompt_tokens=None, completion_tokens=None):
        self.choices = [FakeChoice(FakeMessage(content=content, parsed=parsed))]
        self.usage = FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class GenerateCompletionsStub:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ParseFallbackStub:
    def __init__(self, fallback_content):
        self.fallback_content = fallback_content
        self.parse_calls = []
        self.create_calls = []

        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    parse=self.parse,
                )
            )
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create,
            )
        )

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        raise RuntimeError("structured output failed")

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeResponse(content=self.fallback_content)


class AssignmentNotebookTests(unittest.TestCase):
    def test_notebook_is_present_and_has_expected_cell_count(self):
        notebook = load_notebook()
        self.assertEqual(42, len(notebook["cells"]))

    def test_notebook_contains_strict_prompt_and_analysis_language(self):
        generation_prompt = cell_source(7)
        judge_prompt = cell_source(31)
        analysis_text = cell_source(41)
        analysis_text_lower = analysis_text.lower()

        self.assertIn("Do not invent features, benefits, performance claims, or use cases", generation_prompt)
        self.assertIn("Keep it to exactly one paragraph", generation_prompt)
        self.assertIn("Award good only when the criterion is fully satisfied", judge_prompt)
        self.assertIn("Be conservative", judge_prompt)
        self.assertIn("the remaining real weakness is grounding", analysis_text_lower)
        self.assertIn("far more permissive than the human review", analysis_text_lower)

    def test_final_score_rules_are_strict_and_consistent(self):
        ns = exec_cells([3])
        final_score = ns["final_score"]

        self.assertEqual(
            "PASS",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "good",
                    "tone": "good",
                    "length": "good",
                    "grounding": "ok",
                }
            ),
        )
        self.assertEqual(
            "FAIL",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "good",
                    "tone": "good",
                    "length": "good",
                    "grounding": "bad",
                }
            ),
        )
        self.assertEqual(
            "FAIL",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "good",
                    "tone": "good",
                    "length": "bad",
                    "grounding": "ok",
                }
            ),
        )
        self.assertEqual(
            "",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "",
                    "tone": "good",
                    "length": "good",
                    "grounding": "ok",
                }
            ),
        )
        self.assertEqual(
            "PASS",
            final_score(
                {
                    "fluency": "GOOD",
                    "grammar": " good ",
                    "tone": "Good",
                    "length": "good",
                    "grounding": "ok",
                }
            ),
        )
        self.assertEqual(
            "FAIL",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "ok",
                    "tone": "good",
                    "length": "ok",
                    "grounding": "ok",
                }
            ),
        )
        self.assertEqual(
            "FAIL",
            final_score(
                {
                    "fluency": "good",
                    "grammar": "bad",
                    "tone": "good",
                    "length": "good",
                    "grounding": "good",
                }
            ),
        )

    def test_generation_helpers_calculate_cost_and_ratings(self):
        ns = exec_cells([11], {"pd": pd, "time": time})

        calculate_cost = ns["calculate_cost"]
        rate_latency = ns["rate_latency"]
        rate_cost = ns["rate_cost"]

        self.assertAlmostEqual(
            0.0000153,
            calculate_cost(200, 103, "google/gemma-2-9b-it-fast"),
        )
        self.assertIsNone(calculate_cost(200, 103, "unknown-model"))

        self.assertEqual("good", rate_latency(1299))
        self.assertEqual("ok", rate_latency(1300))
        self.assertEqual("ok", rate_latency(2000))
        self.assertEqual("bad", rate_latency(2001))

        self.assertEqual("good", rate_cost(0.0000148))
        self.assertEqual("ok", rate_cost(0.000016))
        self.assertEqual("bad", rate_cost(0.000018))
        self.assertEqual("", rate_latency(float("nan")))
        self.assertEqual("", rate_cost(float("nan")))

    def test_generate_description_uses_usage_tokens_and_cost(self):
        completions_stub = GenerateCompletionsStub(
            FakeResponse(
                content="A grounded 60-word product description.",
                prompt_tokens=210,
                completion_tokens=90,
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions_stub,
            )
        )
        product_row = {
            "product_name": "Test Product",
            "Product_attribute_list": "features: example feature",
            "material": "aluminum",
            "warranty": "1-year warranty",
        }

        ns = exec_cells(
            [7, 9, 11],
            {"pd": pd, "time": time, "client": client},
        )

        result = ns["generate_description"](product_row)

        self.assertEqual("google/gemma-2-9b-it-fast", result["model"])
        self.assertEqual("A grounded 60-word product description.", result["generated_description"])
        self.assertEqual(210, result["input_tokens"])
        self.assertEqual(90, result["output_tokens"])
        self.assertAlmostEqual(0.0000144, result["cost"])
        self.assertIn("messages", completions_stub.calls[0])
        self.assertEqual("system", completions_stub.calls[0]["messages"][0]["role"])
        self.assertEqual(0.3, completions_stub.calls[0]["temperature"])
        self.assertEqual(120, completions_stub.calls[0]["max_tokens"])
        self.assertEqual("google/gemma-2-9b-it-fast", completions_stub.calls[0]["model"])

    def test_build_user_prompt_includes_all_required_fields(self):
        ns = exec_cells([9])
        prompt = ns["build_user_prompt"](
            {
                "product_name": "Prompt Test",
                "Product_attribute_list": "features: usb-c, oled",
                "material": "glass",
                "warranty": "2-year warranty",
            }
        )

        self.assertIn("Prompt Test", prompt)
        self.assertIn("usb-c, oled", prompt)
        self.assertIn("glass", prompt)
        self.assertIn("2-year warranty", prompt)

    def test_joint_judge_falls_back_to_json_extraction(self):
        fallback_content = """
        ```json
        {
          "Fluency": {"Explanation": "Reads smoothly.", "Verdict": "good"},
          "Grammar": {"Explanation": "No language issues.", "Verdict": "good"},
          "Tone": {"Explanation": "Slightly generic.", "Verdict": "ok"},
          "Length": {"Explanation": "61 words.", "Verdict": "good"},
          "Grounding": {"Explanation": "One mild overstatement.", "Verdict": "ok"}
        }
        ```
        """
        client = ParseFallbackStub(fallback_content)
        row = {
            "product_name": "Judge Test",
            "Product_attribute_list": "features: active noise cancelling",
            "material": "plastic",
            "warranty": "1-year warranty",
            "generated_description": "Some generated description.",
        }

        ns = exec_cells([29, 30, 31, 32, 33], {"client": client})
        result = ns["judge_description"](row)

        self.assertEqual("good", result.fluency.verdict)
        self.assertEqual("ok", result.tone.verdict)
        self.assertEqual("ok", result.grounding.verdict)
        self.assertEqual(1, len(client.parse_calls))
        self.assertEqual(1, len(client.create_calls))
        self.assertEqual("openai/gpt-oss-120b-fast", client.parse_calls[0]["model"])
        self.assertEqual(0, client.parse_calls[0]["temperature"])

    def test_single_criterion_judge_falls_back_to_json_extraction(self):
        fallback_content = """
        **Explanation:**
        {
          "Explanation": "There is one mild overstatement.",
          "Verdict": "ok"
        }
        """
        client = ParseFallbackStub(fallback_content)
        row = {
            "product_name": "Judge Test",
            "Product_attribute_list": "features: active noise cancelling",
            "material": "plastic",
            "warranty": "1-year warranty",
            "generated_description": "Some generated description.",
        }

        ns = exec_cells([29, 30, 31, 32, 33], {"client": client})
        result = ns["judge_single_criterion"](row, "grounding")

        self.assertEqual("ok", result.verdict)
        self.assertIn("mild overstatement", result.explanation)
        self.assertEqual(1, len(client.parse_calls))
        self.assertEqual(1, len(client.create_calls))

    def test_extract_json_text_raises_on_missing_json(self):
        ns = exec_cells([29, 30, 31, 32, 33], {"client": ParseFallbackStub("{}")})
        with self.assertRaises(ValueError):
            ns["extract_json_text"]("no json here")

    def test_improvement_experiment_uses_stricter_decoding_settings(self):
        completions_stub = GenerateCompletionsStub(
            FakeResponse(
                content="An improved grounded description.",
                prompt_tokens=240,
                completion_tokens=80,
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions_stub,
            )
        )
        product_row = {
            "product_name": "Improvement Test",
            "Product_attribute_list": "features: durable, compact",
            "material": "steel",
            "warranty": "2-year warranty",
        }

        ns = exec_cells(
            [9, 11, 22, 23],
            {"pd": pd, "time": time, "client": client},
        )

        result = ns["generate_description_experiment"](product_row)

        self.assertEqual("An improved grounded description.", result["generated_description"])
        self.assertEqual(0.2, completions_stub.calls[0]["temperature"])
        self.assertEqual(110, completions_stub.calls[0]["max_tokens"])
        self.assertIn("Avoid generic hype words", ns["IMPROVED_SYSTEM_PROMPT"])

    def test_task5_results_workbook_contains_expected_sheets(self):
        xl = pd.ExcelFile(TASK5_RESULTS_PATH)
        self.assertEqual(
            [
                "joint_full_run",
                "manual_joint_compare",
                "manual_single_compare",
                "summary",
            ],
            xl.sheet_names,
        )

    def test_assignment_and_task4_workbooks_have_expected_score_distributions(self):
        assignment_df = pd.read_excel(ASSIGNMENT_RESULTS_PATH)
        task4_df = pd.read_excel(TASK4_RESULTS_PATH)

        self.assertEqual({"": 40, "PASS": 5, "FAIL": 5}, assignment_df["final_score"].fillna("").value_counts().to_dict())
        self.assertEqual({"PASS": 7, "FAIL": 3}, task4_df["final_score"].fillna("").value_counts().to_dict())
        self.assertEqual({"ok": 21, "good": 20, "bad": 9}, assignment_df["cost_rating"].fillna("").value_counts().to_dict())
        self.assertEqual({"bad": 6, "ok": 3, "good": 1}, task4_df["cost_rating"].fillna("").value_counts().to_dict())

    def test_task5_workbook_shapes_and_summary_metrics_are_exact(self):
        joint_df = pd.read_excel(TASK5_RESULTS_PATH, sheet_name="joint_full_run")
        manual_joint_df = pd.read_excel(TASK5_RESULTS_PATH, sheet_name="manual_joint_compare")
        manual_single_df = pd.read_excel(TASK5_RESULTS_PATH, sheet_name="manual_single_compare")
        metrics = summary_metrics()

        self.assertEqual((50, 30), joint_df.shape)
        self.assertEqual((10, 30), manual_joint_df.shape)
        self.assertEqual((10, 18), manual_single_df.shape)

        expected_metrics = {
            "manual_rows": 10,
            "full_rows": 50,
            "full_run_judge_model": "openai/gpt-oss-120b-fast",
            "joint_agreement_fluency": 0.9,
            "joint_agreement_grammar": 0.9,
            "joint_agreement_tone": 0.8,
            "joint_agreement_length": 1.0,
            "joint_agreement_grounding": 0.0,
            "single_agreement_fluency": 0.8,
            "single_agreement_grammar": 0.7,
            "single_agreement_tone": 0.6,
            "single_agreement_length": 1.0,
            "single_agreement_grounding": 0.2,
            "human_pass_count_manual": 5,
            "joint_pass_count_manual": 10,
            "single_pass_count_manual": 7,
        }
        self.assertEqual(expected_metrics, metrics)

    def test_task5_outputs_expose_judge_leniency_explicitly(self):
        metrics = summary_metrics()
        self.assertGreater(metrics["joint_agreement_fluency"], metrics["joint_agreement_grounding"])
        self.assertGreater(metrics["joint_pass_count_manual"], metrics["human_pass_count_manual"])
        self.assertGreater(metrics["single_pass_count_manual"], metrics["human_pass_count_manual"])
        self.assertLessEqual(metrics["joint_agreement_grounding"], 0.1)
        self.assertEqual(1.0, metrics["joint_agreement_length"])
        self.assertEqual(1.0, metrics["single_agreement_length"])


if __name__ == "__main__":
    unittest.main()
