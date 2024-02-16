import pandas as pd
from tqdm import tqdm
from flap.api.score import score
from flap.api.match import match


def score_and_match(input_csv, db_path, final_output=None,
                    max_workers=None, in_memory_db=False, classifier_model_path=None,
                    input_address_col='input_address', uprn_col='uprn',
                    score_output=None, score_output_raw=None, score_batch_size=100000,
                    match_output=None, match_output_raw=None, match_batch_size=10000,
                    max_beam_width=200, score_threshold=0.3):

    scored = score(input_csv, db_path, output_file_path=score_output, raw_output_path=score_output_raw,
                   batch_size=score_batch_size, max_workers=max_workers, in_memory_db=in_memory_db,
                   classifier_model_path=classifier_model_path,
                   input_address_col=input_address_col, uprn_col=uprn_col)

    scored['flap_eval_score'] = scored['flap_eval_score'].astype(float)

    scored_to_flap = scored[(scored.flap_eval_score < 0.5) | scored.flap_eval_score.isna()].copy()

    print(len(scored_to_flap))

    scored_skipped = scored[scored.flap_eval_score >= 0.5].copy()
    scored_skipped['flap_match_score'] = scored_skipped['flap_eval_score']
    scored_skipped['flap_uprn'] = scored_skipped['uprn']

    matched = match(input_csv=scored_to_flap, db_path=db_path, output_file_path=match_output,
                    raw_output_path=match_output_raw,
                    batch_size=match_batch_size, max_workers=max_workers, in_memory_db=in_memory_db,
                    classifier_model_path=classifier_model_path,
                    max_beam_width=max_beam_width, score_threshold=score_threshold
                    )

    final = pd.concat([scored_skipped, matched])

    if final_output is not None:
        final.to_csv(final_output)

    return final
