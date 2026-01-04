from typing import List, Union
import re
import random
import numpy as np
import math

from rewards_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

CONTROL_TAG_START = "<control>"
CONTROL_TAG_END = "</control>"
REASON_TAG_START = "<reason>"
REASON_TAG_END = "</reason>"

def semantic_similarity(candidate: str, reference: str) -> float:
    candidate, reference = candidate.strip().lower(), reference.strip().lower()
    if candidate == reference:
        return 1.0
    keywords = ["wait", "hmm", "terminate", "continue", "revise", "switch", "finalize", "reflect"]
    hit = sum(1 for k in keywords if k in candidate and k in reference)
    overlap = len(set(candidate.split()) & set(reference.split()))
    ratio = overlap / (len(reference.split()) + 1e-4)
    return min(1.0, 0.72 + 0.09 * hit + 0.18 * ratio)

def extract_all_control_spans(text: str) -> List[str]:
    return re.findall(r"<control>(.*?)</control>", text, re.DOTALL)

def extract_all_reason_spans(text: str) -> List[str]:
    return re.findall(r"<reason>(.*?)</reason>", text, re.DOTALL)

def structure_score(text: str) -> float:
    c = int(CONTROL_TAG_START in text and CONTROL_TAG_END in text)
    r = int(REASON_TAG_START in text and REASON_TAG_END in text)
    both = c and r
    if both:
        return 1.0
    elif c:
        return 0.6
    elif r:
        return 0.3
    return 0.0

def length_penalty(segment: str, min_words=8, max_words=40, alpha=0.014) -> float:
    l = len(segment.split())
    if l < min_words:
        return max(0.0, 1 - alpha * (min_words - l))
    elif l > max_words:
        return max(0.0, 1 - alpha * (l - max_words))
    return 1.0

def majority_control_vote(control_spans: List[str], refs: List[str], sim_threshold=0.82) -> bool:
    if not control_spans: return False
    sim_matrix = [[semantic_similarity(span, ref) for ref in refs] for span in control_spans]
    votes = sum([any(sim >= sim_threshold for sim in sims) for sims in sim_matrix])
    return votes > (len(control_spans) // 2)

def cluster_equivalent_controls(controls: List[str], refs: List[str], sim_thr=0.82) -> List[str]:
    clusters = []
    for ctrl in controls:
        matched = False
        for cluster in clusters:
            if semantic_similarity(ctrl, cluster[0]) >= sim_thr:
                cluster.append(ctrl)
                matched = True
                break
        if not matched:
            clusters.append([ctrl])
    return [max(cluster, key=len) for cluster in clusters]

def aggregate_reward(sim_scores: List[float], struct_scores: List[float], len_scores: List[float], 
                    w_sim=0.55, w_struct=0.25, w_len=0.2) -> float:
    sim = float(np.mean(sim_scores)) if sim_scores else 0
    struct = float(np.mean(struct_scores)) if struct_scores else 0
    l = float(np.mean(len_scores)) if len_scores else 0
    return w_sim * sim + w_struct * struct + w_len * l

class RewardControlFn(RewardFn):
    def __call__(self, input: RewardInput, ignore_control_token=False, return_detail=False) -> Union[RewardOutput, dict]:
        assert input.problem_type == RewardType.CONTROL
        response = input.model_response
        refs = input.ground_truth.get("control", None)
        if refs is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        if isinstance(refs, (str, float, int)):
            refs = [str(refs)]
        else:
            refs = [str(r) for r in refs]

        controls = extract_all_control_spans(response)
        reasons = extract_all_reason_spans(response)
        if not controls and not ignore_control_token:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        if not controls:
            controls = [response]

        # Cluster equivalent controls if multiple similar
        main_controls = cluster_equivalent_controls(controls, refs)
        sim_scores = []
        len_scores = []
        struct_scores = []

        for ctrl in main_controls:
            sim = max([semantic_similarity(ctrl, ref) for ref in refs])
            sim_scores.append(sim)
            len_scores.append(length_penalty(ctrl))
            struct_scores.append(structure_score(response))

        agg_reward = aggregate_reward(sim_scores, struct_scores, len_scores)
        is_correct = (np.mean(sim_scores) > 0.84) and (np.mean(struct_scores) > 0.6) and (np.mean(len_scores) > 0.65)
        majority = majority_control_vote(main_controls, refs)

        debug = dict(
            reward=agg_reward,
            is_correct=is_correct and majority,
            sim_scores=sim_scores,
            struct_scores=struct_scores,
            len_scores=len_scores,
            main_controls=main_controls,
            reason_spans=reasons,
            majority=majority
        )

        if return_detail:
            return debug
        return RewardOutput(reward=agg_reward, is_correct=is_correct and majority)

