import pandas as pd
import logging
import os
from typing import Dict, Optional
from convokit import Corpus, download

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ADD THESE LINES TO SILENCE NOISE ---
# 1. Silence ConvoKit data warnings
logging.getLogger("convokit").setLevel(logging.ERROR)



class DatasetEnricher:
    """
    Handles the logic of merging local AI response datasets with
    external ground-truth corpora (ConvoKit).
    """

    @staticmethod
    def fetch_cmv_lookup_dict(corpus_name: str = "winning-args-corpus") -> Dict[str, str]:
        """
        Loads the ConvoKit corpus, checking local cache first to avoid re-downloading.
        """
        # 1. Define the default ConvoKit path (Standard location: ~/.convokit/saved-corpora)
        convokit_dir = os.path.expanduser(os.path.join("~", ".convokit", "saved-corpora"))
        corpus_path = os.path.join(convokit_dir, corpus_name)

        # 2. Check if it already exists
        if os.path.exists(corpus_path):
            logger.info(f"Found local corpus at: {corpus_path}")
            data_dir = corpus_path
        else:
            logger.info(f"Corpus not found locally. Downloading '{corpus_name}'...")
            try:
                data_dir = download(corpus_name)
            except Exception as e:
                logger.error(f"Failed to download corpus: {e}")
                raise

        # 3. Load the corpus
        logger.info("Loading corpus into memory...")
        try:
            corpus = Corpus(filename=data_dir)
        except Exception as e:
            logger.error(f"Failed to load corpus from disk: {e}")
            raise

        logger.info("Building ID lookup map...")
        lookup = {}

        # Iterating through conversations to find the root (OP)
        for convo in corpus.iter_conversations():
            try:
                # In CMV corpus, the conversation root is the OP
                op_text = convo.get_utterance(convo.id).text
                lookup[convo.id] = op_text
            except KeyError:
                continue

        logger.info(f"Lookup dictionary ready with {len(lookup)} entries.")
        return lookup

    @staticmethod
    def merge_with_op_data(
            df_local: pd.DataFrame,
            op_lookup: Dict[str, str],
            post_id_col: str = 'post_id'
    ) -> pd.DataFrame:
        """
        Pure function to merge local data with the OP lookup.

        Args:
            df_local: The DataFrame containing AI responses.
            op_lookup: The dictionary of {id: text}.
            post_id_col: The column name in df_local to join on.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'op_text' column.
        """
        if post_id_col not in df_local.columns:
            raise ValueError(f"Column '{post_id_col}' not found in input DataFrame.")

        # Create a copy to avoid SettingWithCopy warnings on the original
        df_enriched = df_local.copy()

        # Map the dictionary
        df_enriched['op_text'] = df_enriched[post_id_col].map(op_lookup)

        # Validation stats
        total = len(df_enriched)
        found = df_enriched['op_text'].notna().sum()
        missing = total - found

        logger.info(f"Merge Complete: {found}/{total} rows matched. ({missing} missing)")

        if found == 0:
            logger.warning("CRITICAL: No rows were matched. Check your post_id format.")

        return df_enriched.dropna(subset=['op_text'])

    @staticmethod
    def enrich_csv(
            input_csv_path: str,
            output_csv_path: Optional[str] = None,
            corpus_name: str = "winning-args-corpus",
            post_id_col: str = 'post_id'
    ) -> pd.DataFrame:
        """
        High-level helper that: loads `input_csv_path`, ensures the `post_id_col` exists,
        fetches the CMV lookup via ConvoKit, merges on `post_id_col`, saves the enriched
        DataFrame to `output_csv_path` (or a default), and returns the enriched DataFrame.

        Args:
            input_csv_path: Path to the input CSV file to enrich.
            output_csv_path: Optional path to write the enriched CSV. If None, a
                             default file next to the input will be created with
                             suffix '_enriched.csv'.
            corpus_name: The ConvoKit corpus name to download/use for lookup.
            post_id_col: Column name in the input CSV that contains the post ids.

        Returns:
            pd.DataFrame: The enriched DataFrame (rows without OP text are dropped).
        """
        # Load input CSV
        logger.info(f"Loading input CSV: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Basic validation
        if post_id_col not in df.columns:
            raise ValueError(f"Column '{post_id_col}' not found in input CSV: {input_csv_path}")

        # Normalize post_id values (strip whitespace and ensure string)
        df[post_id_col] = df[post_id_col].astype(str).str.strip()

        # Fetch lookup from ConvoKit
        lookup = DatasetEnricher.fetch_cmv_lookup_dict(corpus_name)

        # Merge
        enriched = DatasetEnricher.merge_with_op_data(df, lookup, post_id_col=post_id_col)

        # Determine output path
        if output_csv_path is None:
            base, ext = os.path.splitext(input_csv_path)
            output_csv_path = f"{base}_enriched{ext or '.csv'}"

        # Save
        enriched.to_csv(output_csv_path, index=False)
        logger.info(f"Enriched CSV saved to: {output_csv_path} (rows kept: {len(enriched)})")

        return enriched