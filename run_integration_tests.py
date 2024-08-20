import unittest

def main():
    loader = unittest.TestLoader()
    
    combined_suite = unittest.TestSuite()
    test_suite = loader.discover(start_dir='integration', pattern='test_*.py')
    combined_suite.addTests(test_suite)

    runner = unittest.TextTestRunner()
    runner.run(combined_suite)

if __name__ == "__main__":
    main()
