from polquad.utils.bias_calculator import BiasCalculator
from polquad.utils.gemini import GeminiClient

client = GeminiClient()
calculator = BiasCalculator(client)


def test_calibration():
    calculator.calibrate()
    assert calculator.baseline_bias != None

def test_bias_and_magnitude():
    bias, mag = calculator.calculate_bias("Tax cuts for corporations always help the economy.")
    assert bias['x'] != 0.0 or bias['y'] == 0.0 
    assert mag != 0.0


# Test Opinion Agent
    client = GeminiClient()
    agent = OpinionAgent(client, "lib_right")
    sample_statement = "The government spends too much money on social welfare programs."
    
    print(f"TESTING {agent.quadrant.upper()} AGENT")
    print(f"Original statement: {sample_statement}")
    print(f"Revised statement: {agent.provide_opinion(sample_statement)}")

# Judge tests
    from utils.bias_calculator import BiasCalculator
    client = GeminiClient()
    calculator = BiasCalculator(client)
    judge = JudgeAgent(client)

    statement_history = {
        0: {
            "original_statement": "Tax cuts for corporations always help the economy.",
            "bias": {'x': 7.5, 'y': 0.0},
            "magnitude": 7.5
        }
    }

    opinions = {
        "Libertarian Left": {
            "opinion": (
                "Tax cuts for corporations often lead to increased wealth concentration "
                "rather than broad economic benefit."
            ),
            "bias": {'x': 3.5, 'y': 0.0},
            "magnitude": 3.5
        },
        "Libertarian Right": {
        "opinion": (
            "Tax cuts for corporations can stimulate economic growth by encouraging "
            "investment and job creation."
        ),
        "bias": {'x': 6.5, 'y': 2.0},
            "magnitude": 6.80073525
        },
        "Authoritarian Left": {
            "opinion": (
                "Tax cuts for corporations disproportionately benefit shareholders "
                "and executives, diverting resources that could be invested in "
                "public services and worker wages, ultimately hindering broad-based "
                "economic prosperity."
            ),
            "bias": {'x': -4.5, 'y': 0.0},
            "magnitude": 4.5
        },
        "Authoritarian Right": {
            "opinion": (
                "Tax cuts for corporations, when strategically implemented, can "
                "foster economic growth and job creation."
            ),
            "bias": {'x': 6.5, 'y': 2.0},
            "magnitude": 6.80073525
        }
    }

    print(f"TESTING JUDGE AGENT")
    print(f"Original Statement: {statement_history[0]['original_statement']}")
    print(f"Original Bias: {statement_history[0]['bias']}")
    print(f"Original Bias Magnitude: {statement_history[0]['magnitude']}")

    moderated_statement = judge.neutralize_opinions(opinions, statement_history)
    moderated_bias, moderated_mag = calculator.calculate_bias(moderated_statement)

    print(f"Moderated Statement: {moderated_statement}")
    print(f"Moderated Statement Bias: {moderated_bias}")
    print(f"Moderated Bias Magnitude: {moderated_mag}")


# Test Unified Agent
    client = GeminiClient()
    from utils.bias_calculator import BiasCalculator
    calculator = BiasCalculator(client)
    agent = UnifiedAgent(client)
    sample_statement = "The government spends too much money on social welfare programs."
    
    statement_history = {
        0: {
            "original_statement": "Tax cuts for corporations always help the economy.",
            "bias": {'x': 7.5, 'y': 0.0},
            "magnitude": 7.5
        }
    }

    print(f"TESTING UNIFIED AGENT")
    print(f"Original Statement: {statement_history[0]['original_statement']}")
    print(f"Original Bias: {statement_history[0]['bias']}")
    print(f"Original Bias Magnitude: {statement_history[0]['magnitude']}")

    moderated_statement = agent.neutralize(statement_history)
    moderated_bias, moderated_mag = calculator.calculate_bias(moderated_statement)

    print(f"Moderated Statement: {moderated_statement}")
    print(f"Moderated Statement Bias: {moderated_bias}")
    print(f"Moderated Bias Magnitude: {moderated_mag}")