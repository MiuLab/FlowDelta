import collections
from collections import Counter, defaultdict
import math
import numpy as np
import json
 
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from CoQA_eval import CoQAEvaluator

import re, string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
def handle_cannot(refs):
    num_cannot = 0
    num_spans = 0
    for ref in refs:
        if ref == 'CANNOTANSWER':
            num_cannot += 1
        else:
            num_spans += 1
    if num_cannot >= num_spans:
        refs = ['CANNOTANSWER']
    else:
        refs = [x for x in refs if x != 'CANNOTANSWER']
    return refs

def leave_one_out(refs):
    if len(refs) == 1:
        return 1.
    splits = []
    for r in refs:
        splits.append(r.split())
    t_f1 = 0.0
    for i in range(len(refs)):
        m_f1 = 0
        for j in range(len(refs)):
            if i == j:
                continue
            f1_ij = f1_score(refs[i], refs[j])
            if f1_ij > m_f1:
                m_f1 = f1_ij
        t_f1 += m_f1
    return t_f1 / len(refs)

def leave_one_out_max(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        scores_for_ground_truths.append(single_score(prediction, ground_truth))

    if len(scores_for_ground_truths) == 1:
        return scores_for_ground_truths[0]
    else:
        # leave out one ref every time
        t_f1 = []
        for i in range(len(scores_for_ground_truths)):
            t_f1.append(max(scores_for_ground_truths[:i] + scores_for_ground_truths[i+1:]))
        return 1.0 * sum(t_f1) / len(t_f1)

def single_score(prediction, ground_truth):
    if prediction == "CANNOTANSWER" and ground_truth == "CANNOTANSWER":
        return 1.0
    elif prediction == "CANNOTANSWER" or ground_truth == "CANNOTANSWER":
        return 0.0
    else:
        return f1_score(prediction, ground_truth)

def quac_performance(prediction, nbest_pred, target_dict):
    pred, truth = [], []

    for qa_id, span in prediction.items():
        dialog_id = qa_id.split("_q#")[0]
        
        if len(span) == 0:
            span = 'CANNOTANSWER'
        
        pred.append(span)
        truth.append(target_dict[qa_id])
    min_F1 = 0.4
    clean_pred, clean_truth = [], []

    all_f1 = []
    for p, t in zip(pred, truth):
        clean_t = handle_cannot(t)
        # compute human performance
        human_F1 = leave_one_out(clean_t)
        if human_F1 < min_F1: continue

        clean_pred.append(p)
        clean_truth.append(clean_t)
        all_f1.append(leave_one_out_max(p, clean_t))

    cur_f1, best_f1 = sum(all_f1), sum(all_f1)

    return 100.0 * best_f1 / len(clean_pred)

def coqa_performance(prediction, nbest_pred, evaluator):
    pred = {}
    class_preds = []
    for qa_id, span in prediction.items():
        passage_id, turn_id = qa_id.split('#')
        key = (passage_id, int(turn_id))
        pred[key] = span
    
    result = evaluator.model_performance(pred)
    return result['overall']['f1']
    

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def dev_evaluate(all_examples, all_features, all_results, n_best_size,
                 max_answer_length, do_lower_case, verbose_logging, is_version2, null_score_diff_threshold, 
                 dataset='coqa', evaluator=None):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "class_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    class_score_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if is_version2:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            class_logit=result.class_logits))

        if is_version2:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    class_logit=0))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "class_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    class_logit=pred.class_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if is_version2:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="CANNOTANSWER", start_logit=null_start_logit,
                        end_logit=null_end_logit, class_logit=0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="CANNOTANSWER", start_logit=0.0, end_logit=0.0, class_logit=0))

        assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        
        if not best_non_null_entry:
            best_non_null_entry = nbest[0]
        
        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output['class_logit'] = entry.class_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        
        if not is_version2:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            #if score_diff > null_score_diff_threshold:
            #    all_predictions[example.qas_id] = ""
            #else:
            #    all_predictions[example.qas_id] = best_non_null_entry.text
            all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    if dataset == 'coqa':
        return coqa_performance(all_predictions, all_nbest_json, evaluator)
    elif dataset == 'quac':
        return quac_performance(all_predictions, all_nbest_json, evaluator)
    else:
        assert False
        
def write_quac(prediction, nbest_pred, in_file, out_file):
    dialog_pred = defaultdict(list)
    for qa_id, span in prediction.items():
        dialog_id = qa_id.split("_q#")[0]

        if len(span) == 0:
            span = 'CANNOTANSWER'
        dialog_pred[dialog_id].append([qa_id, span])

    # for those we don't predict anything
    target = json.load(open(in_file))['data']
    for p in target:
        for par in p['paragraphs']:
            dialog_id = par['id']
            qa_list = par['qas']
            if dialog_id not in dialog_pred:
                for qa in qa_list:
                    qa_id = qa['id']
                    dialog_pred[dialog_id].append([qa_id, 'CANNOTANSWER'])

    # now we predict
    with open(out_file, 'w') as fout:
        for dialog_id, dialog_span in dialog_pred.items():
            output_dict = {'best_span_str': [], 'qid': [], 'yesno':[], 'followup': []}
            for qa_id, span in dialog_span:
                output_dict['best_span_str'].append(span)
                output_dict['qid'].append(qa_id)
                output_dict['yesno'].append('y')
                output_dict['followup'].append('y')
            fout.write(json.dumps(output_dict) + '\n')

def write_coqa(prediction, nbest_pred, in_file, out_file):
    pred = {}
    class_preds = []
    for qa_id, span in prediction.items():
        passage_id, turn_id = qa_id.split('#')
        key = (passage_id, int(turn_id))
        pred[key] = span

    
    target = json.load(open(in_file))['data']
    for p in target:
        pid = p['id']
        for q in p['questions']:
            turn_id = q['turn_id']
            key = (pid, turn_id)
            if key not in pred:
                pred[key] = 'unknown'
    # make 
    official_pred = []
    for key, p in pred.items():
        official_pred.append({'id': key[0],
                             'turn_id': key[1],
                             'answer': p})
    with open(out_file, 'w') as fout:
        json.dump(official_pred, fout)
