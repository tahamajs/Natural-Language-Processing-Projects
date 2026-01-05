import unittest
import os
from src.analyzer import calculate_total_revenue, get_average_transaction, get_maximum_transaction
from src.file_handler import load_transactions

class TestLegacySystem(unittest.TestCase):
    
    def setUp(self):
        
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.data_file = os.path.join(base_path, 'data', 'transactions.csv')
        self.empty_file = os.path.join(base_path, 'data', 'empty.csv')

    def test_load_transactions(self):
        transactions = load_transactions(self.data_file)
        self.assertGreater(len(transactions), 0)

    def test_calculate_total_revenue(self):
        transactions = load_transactions(self.data_file)
        revenue = calculate_total_revenue(transactions)
        self.assertAlmostEqual(revenue, 1350.50, places=2)

    def test_get_average_transaction(self):
        transactions = load_transactions(self.data_file)
        avg = get_average_transaction(transactions)
        self.assertAlmostEqual(avg, 337.625, places=2)

    def test_get_maximum_transaction(self):
        transactions = load_transactions(self.data_file)
        max_transaction = get_maximum_transaction(transactions)
        self.assertAlmostEqual(max_transaction, 1000.00, places=2)  # Corrected expected max

    def test_handle_empty_file(self):
        
        transactions = load_transactions(self.empty_file)
        self.assertEqual(len(transactions), 0)
        
        revenue = calculate_total_revenue(transactions)
        self.assertEqual(revenue, 0.0)

        avg = get_average_transaction(transactions)
        self.assertEqual(avg, 0.0)

        max_transaction = get_maximum_transaction(transactions)
        self.assertEqual(max_transaction, 0.0)

if __name__ == '__main__':
    unittest.main()
