import unittest
import upsaclay_ml

class TestCore(unittest.TestCase):
    def test_import(self):
        """Test that the package can be imported."""
        self.assertIsNotNone(upsaclay_ml)
        
    def test_version_exists(self):
        """Test that version is defined."""
        self.assertTrue(hasattr(upsaclay_ml, '__version__'))
        
if __name__ == '__main__':
    unittest.main()
