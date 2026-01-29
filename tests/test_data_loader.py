import unittest
import pandas as pd
from scripts.data_loader import DatasetEnricher


class TestDatasetEnricher(unittest.TestCase):
    def test_merge_with_op_data_basic(self):
        mock_local_data = pd.DataFrame([
            {'post_id': 't3_12345', 'text': 'I disagree.'},
            {'post_id': 't3_99999', 'text': 'Wrong ID.'},
        ])

        mock_lookup = {
            't3_12345': 'I believe the sky is green.',
            't3_67890': 'Unused OP text.'
        }

        result_df = DatasetEnricher.merge_with_op_data(mock_local_data, mock_lookup)

        # Only one row should remain (the one with a matching post_id)
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]['op_text'], 'I believe the sky is green.')


if __name__ == '__main__':
    unittest.main()

