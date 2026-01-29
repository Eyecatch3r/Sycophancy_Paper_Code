import unittest
import tempfile
import os
import sys
import importlib.util
import pandas as pd
from unittest.mock import patch

# Ensure project root is on sys.path when tests are run from tests/ directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Load the CLI module by file location to avoid package import quirks
_cli_path = os.path.join(PROJECT_ROOT, 'scripts', 'enrich_csv_cli.py')
spec = importlib.util.spec_from_file_location('enrich_csv_cli', _cli_path)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class TestEnrichCsvCLI(unittest.TestCase):
    def test_cli_enriches_csv_using_mocked_lookup(self):
        # Create temporary input CSV
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')

        df = pd.DataFrame([
            {'post_id': 't3_1', 'text': 'a'},
            {'post_id': 't3_2', 'text': 'b'},
            {'post_id': 't3_3', 'text': 'c'},  # will be missing in lookup
        ])
        df.to_csv(input_path, index=False)

        # Provide a mocked lookup dict (no network)
        mock_lookup = {'t3_1': 'op one', 't3_2': 'op two'}

        # Patch DatasetEnricher.fetch_cmv_lookup_dict to return our mock_lookup
        with patch('data_loader.DatasetEnricher.fetch_cmv_lookup_dict', return_value=mock_lookup) as mock_fetch:
            # Run CLI which will call enrich_csv and write output_path
            returned_df = cli.main(['--input', input_path, '--output', output_path, '--corpus', 'dummy'])

            # Ensure the fetch was called
            mock_fetch.assert_called_once()

        # Assert output file was created
        self.assertTrue(os.path.exists(output_path), f"Expected output CSV at {output_path}")

        out_df = pd.read_csv(output_path)
        # Should have dropped the unmatched 't3_3'
        self.assertEqual(len(out_df), 2)
        self.assertIn('op_text', out_df.columns)

        # Check that op_text values match the mock lookup in both returned df and file
        mapping_file = dict(zip(out_df['post_id'], out_df['op_text']))
        mapping_returned = dict(zip(returned_df['post_id'], returned_df['op_text']))
        self.assertEqual(mapping_file, mapping_returned)
        self.assertEqual(mapping_file['t3_1'], 'op one')
        self.assertEqual(mapping_file['t3_2'], 'op two')


if __name__ == '__main__':
    unittest.main()
