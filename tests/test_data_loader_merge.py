import unittest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch


current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (Sycophancy_Paper_Code)
project_root = os.path.dirname(current_dir)
# Add root to sys.path
sys.path.insert(0, project_root)

from scripts.data_loader import DatasetEnricher


class TestDatasetEnricherMerge(unittest.TestCase):
    def test_merge_with_op_data_happy_path(self):
        df = pd.DataFrame([
            {'post_id': 't3_1', 'text': 'a'},
            {'post_id': 't3_2', 'text': 'b'},
            {'post_id': 't3_3', 'text': 'c'},
        ])
        lookup = {'t3_1': 'op one', 't3_2': 'op two'}

        out = DatasetEnricher.merge_with_op_data(df, lookup, post_id_col='post_id')

        self.assertEqual(len(out), 2)
        self.assertListEqual(sorted(out['post_id'].tolist()), ['t3_1', 't3_2'])
        self.assertIn('op_text', out.columns)
        self.assertEqual(out.loc[out['post_id'] == 't3_1', 'op_text'].iloc[0], 'op one')

    def test_merge_with_op_data_missing_column_raises(self):
        df = pd.DataFrame([
            {'id': 't3_1', 'text': 'a'},
        ])
        lookup = {'t3_1': 'op one'}

        with self.assertRaises(ValueError):
            DatasetEnricher.merge_with_op_data(df, lookup, post_id_col='post_id')

    def test_merge_with_op_data_no_matches_returns_empty(self):
        df = pd.DataFrame([
            {'post_id': 't3_9', 'text': 'x'},
            {'post_id': 't3_10', 'text': 'y'},
        ])
        lookup = {'t3_1': 'op one'}

        out = DatasetEnricher.merge_with_op_data(df, lookup, post_id_col='post_id')
        self.assertEqual(len(out), 0)

    def test_enrich_csv_normalizes_post_id_and_saves(self):
        # ensure enrich_csv strips whitespace from post_id when matching
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')

        df = pd.DataFrame([
            {'post_id': ' t3_1 ', 'text': 'a'},
            {'post_id': '\tt3_2\n', 'text': 'b'},
        ])
        df.to_csv(input_path, index=False)

        # Provide a mocked lookup dict (no network)
        mock_lookup = {'t3_1': 'op one', 't3_2': 'op two'}

        # --- FIX: PATCH STRING MUST MATCH THE IMPORT PATH ---
        # Was: 'data_loader.DatasetEnricher...'
        # Now: 'scripts.data_loader.DatasetEnricher...'
        with patch('scripts.data_loader.DatasetEnricher.fetch_cmv_lookup_dict', return_value=mock_lookup):
            enriched = DatasetEnricher.enrich_csv(input_path, output_csv_path=output_path, corpus_name='dummy')

        # Check file exists and normalization led to matches
        self.assertTrue(os.path.exists(output_path))
        out_df = pd.read_csv(output_path)
        self.assertEqual(len(out_df), 2)
        self.assertIn('op_text', out_df.columns)
        mapping = dict(zip(out_df['post_id'], out_df['op_text']))
        self.assertEqual(mapping['t3_1'], 'op one')
        self.assertEqual(mapping['t3_2'], 'op two')


if __name__ == '__main__':
    unittest.main()