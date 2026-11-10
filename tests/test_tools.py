"""Unit tests for the mock tool. No API calls. Run: .venv/bin/python -m unittest discover -s tests -v"""
import unittest

from support_agent.brain.tools import check_transfer_status, normalize_transfer_id


class NormalizeTests(unittest.TestCase):
    def test_spaces_and_dashes_removed(self):
        self.assertEqual(normalize_transfer_id("1 2 3 4 5"), "12345")
        self.assertEqual(normalize_transfer_id("123-45"), "12345")

    def test_rejects_bad_input(self):
        for bad in (None, "", "abc", "1234", "1234567890123", "12a45"):
            self.assertIsNone(normalize_transfer_id(bad), bad)


class LookupTests(unittest.TestCase):
    def test_found(self):
        r = check_transfer_status("12345")
        self.assertEqual(r["status"], "Transfer Sent")
        self.assertEqual(r["transfer_id"], "12345")

    def test_found_with_spoken_spacing(self):
        self.assertEqual(check_transfer_status("6 7 8 9 0")["transfer_id"], "67890")

    def test_not_found(self):
        self.assertEqual(check_transfer_status("99999")["error"], "not_found")

    def test_invalid_format(self):
        self.assertEqual(check_transfer_status("12")["error"], "invalid_format")
        self.assertEqual(check_transfer_status(None)["error"], "invalid_format")

    def test_all_five_states_exist(self):
        for tid in ("12345", "67890", "24680", "13579", "11223"):
            self.assertNotIn("error", check_transfer_status(tid))


if __name__ == "__main__":
    unittest.main()
