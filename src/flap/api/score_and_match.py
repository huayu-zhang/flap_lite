import pandas as pd
from tqdm import tqdm
from flap.api.score import score
from flap.api.match import match


def score_and_match(input_csv, db_path, final_output=None,
                    max_workers=None, in_memory_db=False, classifier_model_path=None,
                    input_address_col='input_address', uprn_col='uprn',
                    score_output=None, score_output_raw=None, score_batch_size=100000,
                    match_output=None, match_output_raw=None, match_batch_size=10000,
                    match_in_progress_log_path=None, match_max_log_interval=4800,
                    max_beam_width=200, score_threshold=0.3):

    """
    This is the workflow to score existing matches and match the remaining ones and ones with scores below 0.5

    Parameters
    ----------
    input_csv : str, or pandas.DataFrame
        Path to the csv file. The file needs to have two fields in the header ['input_id', 'input_address']
    db_path : str
        Path to the database built. See `flap.create_db()`
    final_output: str, default None
        Path for saving the final output csv file, containing ['input_id', 'input_address', 'uprn', 'score'].
        If None, results are not saved
    max_workers : int, default None
        Number of processes. If None, the max cpu available is determined by `flap.utils.cpu_count.available_cpu_count()`
    in_memory_db : bool, default False
        If in-memory SQLite database is used. If True, a temp database is created in shared memory cache from pre-built
        csv files
    classifier_model_path : str, default None
        The path to the pretrained sklearn classifier model.
        If None, the model is loaded from 'flap.__file__/model/*.clf'
    input_address_col: str, default 'input_address'
        The column name for input addresses for score function
    uprn_col: str, default 'uprn'
        The column name for input UPRN for score function
    score_output
    score_output_raw
    score_batch_size
    match_output
    match_output_raw
    match_batch_size
    match_in_progress_log_path
    match_max_log_interval
    max_beam_width: int, default 200
        The max number of rows to be considered from UPRN database
    score_threshold: float, default 0.3
        The min score for early stop of matching

    Returns
    -------
    pandas.DataFrame
        Final results
    """

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
                    in_progress_log_path=match_in_progress_log_path, max_log_interval=match_max_log_interval,
                    batch_size=match_batch_size, max_workers=max_workers, in_memory_db=in_memory_db,
                    classifier_model_path=classifier_model_path,
                    max_beam_width=max_beam_width, score_threshold=score_threshold
                    )

    final = pd.concat([scored_skipped, matched])

    if final_output is not None:
        final.to_csv(final_output)

    return final
