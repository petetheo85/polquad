import pytest
from math import sqrt
from polquad.utils.bias_calculator import BiasCalculator
from polquad.agents.opinion_agent import OpinionAgent
from polquad.agents.judge_agent import JudgeAgent
from polquad.agents.naive_agent import NaiveAgent
from polquad.agents.unified_agent import UnifiedAgent
from polquad.frameworks.naive import NaiveFramework
from polquad.frameworks.polquad import PolquadFramework
from polquad.frameworks.unified import UnifiedPolquadFramework
from polquad.utils.gemini_client import GeminiClient
from polquad.utils.openai_client import ChatGPTClient
from polquad.utils.claude_client import ClaudeClient


class MockLLMClient:
    """Deterministic, offline mock client simulating LLM responses for test assertions."""

    def __init__(self, api_key: str = "mock_key", model: str = "mock-model"):
        self.api_key = api_key
        self.model = model

    def generate_text(self, prompt: str, system_instruction: str = None, temperature: float = 0.7, max_output_tokens: int = 1024) -> str:
        instruction = system_instruction or ""
        if "apolitical AI judge" in instruction:
            return "Tax policy should balance market competitiveness with equitable social investment."
        if "Libertarian Left" in instruction:
            return "Tax policy must prevent wealth concentration and fund public infrastructure."
        if "Authoritarian Right" in instruction:
            return "Tax policy should preserve national industry and maintain legal economic order."
        if "neutral and objective AI editor" in instruction:
            return "Tax policies involve trade-offs between economic efficiency and public welfare."
        return "A balanced statement regarding fiscal policy and governance."

    def generate_json(self, prompt: str, system_instruction: str = None, temperature: float = 0.0) -> dict:
        if "The sky is blue" in prompt:
            return {"x": 0.0, "y": 0.0}
        if any(term in prompt for term in ["balance", "trade-offs", "equitable"]):
            return {"x": 0.2, "y": 0.1}
        return {"x": 2.0, "y": -1.5}


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def test_config():
    return {
        "max_iters": 2,
        "bias_threshold": 1.0,
        "bias_calc_runs": 1,
        "verbose": False
    }


class TestBiasCalculator:
    """Unit tests for coordinate extraction, calibration, and distance metrics."""

    def test_calibration(self, mock_client, test_config):
        calculator = BiasCalculator(mock_client, test_config)
        assert calculator.baseline_bias == {"x": 0.0, "y": 0.0}

    def test_magnitude(self, mock_client, test_config):
        calculator = BiasCalculator(mock_client, test_config)
        bias, mag = calculator.calculate_bias("Sample fiscal statement.")
        assert bias["x"] == 2.0
        assert bias["y"] == -1.5
        expected_magnitude = sqrt(2.0**2 + (-1.5)**2)
        assert abs(mag - expected_magnitude) < 1e-5


class TestAgents:
    """Unit tests for individual agent personas and iterative neutralization."""

    def test_opinion_agent(self, mock_client):
        agent = OpinionAgent(mock_client, "lib_left")
        opinion = agent.provide_opinion("Tax cuts for corporations help the economy.")
        assert isinstance(opinion, str)
        assert len(opinion) > 0
        assert "public infrastructure" in opinion

    def test_judge_agent_initial_and_iterative(self, mock_client):
        judge = JudgeAgent(mock_client)
        history = {
            0: {
                "original_statement": "Tax cuts for corporations help the economy.",
                "bias": {"x": 5.0, "y": 0.0},
                "magnitude": 5.0
            }
        }
        opinions = {
            "lib_left": {"opinion": "Prevents public funding.", "bias": {"x": -3.0, "y": 0.0}, "magnitude": 3.0}
        }

        # Initial call
        neutralized_initial = judge.neutralize_opinions(opinions, history, bias_threshold=1.0, initial_call=True)
        assert isinstance(neutralized_initial, str)
        assert "Tax policy" in neutralized_initial

        # Iterative refinement call with previous attempt recorded
        history[1] = {
            "moderated_statement": neutralized_initial,
            "bias": {"x": 1.5, "y": 0.2},
            "magnitude": 1.51,
            "converged": False
        }
        neutralized_iter = judge.neutralize_opinions(opinions, history, bias_threshold=1.0, initial_call=False)
        assert isinstance(neutralized_iter, str)
        assert len(neutralized_iter) > 0

    def test_naive_agent_modes(self, mock_client):
        agent = NaiveAgent(mock_client)
        history = {
            0: {
                "original_statement": "Extreme deregulation is always beneficial.",
                "bias": {"x": 6.0, "y": -2.0},
                "magnitude": 6.32
            }
        }

        # Initial neutralization
        first_pass = agent.neutralize(history, bias_threshold=1.0, initial_call=True)
        assert isinstance(first_pass, str)
        assert len(first_pass) > 0

        # Second pass incorporating historical feedback
        history[1] = {
            "moderated_statement": first_pass,
            "bias": {"x": 2.1, "y": -0.8},
            "magnitude": 2.25,
            "converged": False
        }
        second_pass = agent.neutralize(history, bias_threshold=1.0, initial_call=False)
        assert isinstance(second_pass, str)
        assert len(second_pass) > 0

    def test_unified_agent_modes(self, mock_client):
        agent = UnifiedAgent(mock_client)
        history = {
            0: {
                "original_statement": "Only total central planning can deliver social justice.",
                "bias": {"x": -8.0, "y": 7.0},
                "magnitude": 10.63
            }
        }

        # Initial call
        first_pass = agent.neutralize(history, bias_threshold=1.0, initial_call=True)
        assert isinstance(first_pass, str)
        assert len(first_pass) > 0

        # Iterative call
        history[1] = {
            "moderated_statement": first_pass,
            "bias": {"x": -2.5, "y": 1.8},
            "magnitude": 3.08,
            "converged": False
        }
        second_pass = agent.neutralize(history, bias_threshold=1.0, initial_call=False)
        assert isinstance(second_pass, str)
        assert len(second_pass) > 0


class TestFrameworks:
    """Integration tests for orchestrator loops, convergence, and metric calculation."""

    def test_instantiations(self, mock_client, test_config):
        calculator = BiasCalculator(mock_client, test_config)
        polquad_fw = PolquadFramework(config=test_config, bias_calculator=calculator, client=mock_client)
        unified_fw = UnifiedPolquadFramework(config=test_config, bias_calculator=calculator, client=mock_client)
        naive_fw = NaiveFramework(config=test_config, bias_calculator=calculator, client=mock_client)

        assert polquad_fw.frame_work_name == "full_polquad"
        assert unified_fw.frame_work_name == "unified_polquad"
        assert naive_fw.frame_work_name == "naive"

    def test_full_moderation_cycle(self, mock_client, test_config):
        calculator = BiasCalculator(mock_client, test_config)
        polquad_fw = PolquadFramework(config=test_config, bias_calculator=calculator, client=mock_client)

        sample_row = {
            "text": "Subsidies must be eliminated immediately across all sectors.",
            "quadrant": "Libertarian Right"
        }
        initial_coords = {"x": 6.0, "y": -4.0}
        initial_mag = sqrt(6.0**2 + (-4.0)**2)

        result = polquad_fw.run_analysis(index=0, row=sample_row, initial_bias_coords=initial_coords, initial_bias_mag=initial_mag)

        assert result["index"] == 0
        assert result["original_statement"] == sample_row["text"]
        assert result["true_label"] == "Libertarian Right"

        fw_result = result["frameworks"]["full_polquad"]
        assert "opinions" in fw_result
        assert len(fw_result["opinions"]) == 4
        assert "moderation_history" in fw_result
        assert fw_result["num_iterations"] >= 1
        assert fw_result["converged"] is True
        assert fw_result["bias_reduction"] > 0

    def test_early_exit_when_already_neutral(self, mock_client, test_config):
        calculator = BiasCalculator(mock_client, test_config)
        polquad_fw = PolquadFramework(config=test_config, bias_calculator=calculator, client=mock_client)

        sample_row = {"text": "A balanced neutral statement with negligible political slant."}
        initial_coords = {"x": 0.3, "y": 0.2}
        initial_mag = sqrt(0.3**2 + 0.2**2)

        result = polquad_fw.run_analysis(index=1, row=sample_row, initial_bias_coords=initial_coords, initial_bias_mag=initial_mag)
        fw_result = result["frameworks"]["full_polquad"]

        assert fw_result["num_iterations"] == 0
        assert fw_result["final_bias_magnitude"] == initial_mag


class TestLLMClients:
    """Verifies client wrapper initialization, key handling, and validation without network access."""

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        # Ensure environment variables are clear
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY not found"):
            GeminiClient()

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY not found"):
            ChatGPTClient()

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not found"):
            ClaudeClient()

    def test_explicit_api_key_initialization(self):
        # Ensure clients accept an explicitly passed API key without reading environment
        gemini = GeminiClient(api_key="test-gemini-key")
        assert gemini.api_key == "test-gemini-key"

        openai_client = ChatGPTClient(api_key="test-openai-key")
        assert openai_client.api_key == "test-openai-key"

        claude = ClaudeClient(api_key="test-claude-key")
        assert claude.api_key == "test-claude-key"