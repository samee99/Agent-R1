import numpy as np
import json

EXPECTED_SPECTRUM_LENGTH = 57

def parse_spectrum_from_tool_response(tool_response_str):
    try:
        tool_response = json.loads(tool_response_str)
        spectrum = tool_response.get("content", {}).get("spectrum", None)
        if spectrum is None or not isinstance(spectrum, list):
            raise ValueError("Missing or malformed 'spectrum' in tool response")
        if len(spectrum) != EXPECTED_SPECTRUM_LENGTH:
            raise ValueError(f"Spectrum length mismatch: got {len(spectrum)}, expected {EXPECTED_SPECTRUM_LENGTH}")
        return spectrum
    except Exception as e:
        print(f"⚠️ Failed to parse spectrum: {e}")
        return [0.0] * EXPECTED_SPECTRUM_LENGTH

def compute_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if len(v1) != len(v2):
        min_len = min(len(v1), len(v2))
        print(f"⚠️ Truncating spectra to {min_len} for cosine similarity")
        v1 = v1[:min_len]
        v2 = v2[:min_len]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

def _default_compute_score_format(data_source, solution_str, extra_info=None):
    if data_source == "layer_stack":
        return 1.0
    elif data_source in ['hotpotqa/hotpot_qa', 'bdsaglam/musique', 'xanhho/2WikiMultihopQA']:
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format(solution_str)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format(solution_str)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format(solution_str)
    else:
        raise NotImplementedError
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])

def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "layer_stack":
        predicted = [x.strip() for x in solution_str.split(",")]
        return float(predicted == ground_truth)
    elif data_source in ['hotpotqa/hotpot_qa', 'bdsaglam/musique', 'xanhho/2WikiMultihopQA']:
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_answer(solution_str, ground_truth)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])

def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "layer_stack":
        spectrum = parse_spectrum_from_tool_response(solution_str)
        target_spectrum = extra_info.get("target_spectrum", None)
        if target_spectrum is None:
            raise ValueError("Missing target_spectrum in extra_info for layer_stack")
        return compute_cosine_similarity(spectrum, target_spectrum)
    elif data_source in ['hotpotqa/hotpot_qa', 'bdsaglam/musique', 'xanhho/2WikiMultihopQA']:
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info),
        "acc": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
    }
