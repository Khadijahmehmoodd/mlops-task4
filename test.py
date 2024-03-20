import unittest
import subprocess

class TestApp(unittest.TestCase):
    def test_app_script(self):
        # Check if the app.py script can be executed without errors
        try:
            subprocess.run(['python', 'app.py'], check=True)
        except subprocess.CalledProcessError as e:
            self.fail(f"Running app.py script failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
