#!/usr/bin/env python3

import unittest
from services.ocr.ocr_service import OCRService
from config.ocr_config import OCRConfig


class TestOCRScoring(unittest.TestCase):
    """Test cases for OCR scoring logic validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.ocr_service = OCRService(config=OCRConfig(debug=False, debug_ocr=False))

    def test_good_ocr_text_high_score(self):
        """Test that good OCR text receives high scores"""
        good_texts = [
            # Perfect receipt with all components
            """GROCERY STORE RECEIPT
APPLES 3.99
ORANGES 2.50
BANANAS 1.25
MILK 4.99
BREAD 2.99
SUBTOTAL 15.72
TAX 1.26
TOTAL 16.98
CASH 16.98
THANK YOU FOR SHOPPING WITH US TODAY
HAVE A GREAT DAY""",

            # Long receipt with many items
            """SUPERMARKET RECEIPT
GROCERY ITEMS
MILK 4.99 1 GAL
BREAD 2.99 LOAF
EGGS 3.99 DOZEN
CHEESE 5.99 LB
BUTTER 4.49 LB
YOGURT 1.99 6PK
CEREAL 3.99 BOX
COFFEE 7.99 BAG
TEA 3.49 BOX
JUICE 4.99 64OZ
PRODUCE
APPLES 1.99 LB
BANANAS 0.69 LB
ORANGES 1.49 LB
TOMATOES 2.99 LB
LETTUCE 2.49 HEAD
CARROTS 1.99 BAG
MEAT
CHICKEN 8.99 LB
BEEF 12.99 LB
PORK 7.99 LB
FROZEN FOODS
ICE CREAM 5.99 QT
FROZEN VEGETABLES 2.99 BAG
FROZEN PIZZA 7.99 EACH
DAIRY
SOUR CREAM 2.99 16OZ
CREAM CHEESE 3.99 8OZ
BUTTER 4.99 LB
SNACKS
CHIPS 3.99 BAG
COOKIES 2.99 PACKAGE
CANDY 1.99 BAG
NUTS 4.99 CAN
BEVERAGES
SODA 5.99 12PK
WATER 3.99 CASE
JUICE 4.99 6PK
ENERGY DRINK 2.99 CAN
CLEANING SUPPLIES
SOAP 3.99 BAR
SHAMPOO 5.99 BOTTLE
PAPER TOWELS 8.99 ROLL
TRASH BAGS 4.99 BOX
SUBTOTAL 125.67
TAX 10.05
TOTAL 135.72
PAYMENT METHOD: CREDIT CARD
AUTHORIZATION: 123456
THANK YOU FOR YOUR PURCHASE
PLEASE COME AGAIN SOON""",

            # Receipt with detailed formatting
            """WALMART SUPERCENTER
RECEIPT #12345
DATE: 03/15/2024 14:32:15
CASHIER: #4567 JOHN D
STORE #7890

GROCERY DEPARTMENT
ORGANIC BANANAS 0.69 LB @ 0.69/LB 0.69
ORGANIC APPLES 1.99 LB @ 1.99/LB 1.99
FRESH SALMON 12.99 LB @ 12.99/LB 12.99
CHICKEN BREAST 4.99 LB @ 4.99/LB 4.99

PRODUCE DEPARTMENT
ROMAINE LETTUCE 2.49 EACH 2.49
TOMATOES 2.99 LB @ 2.99/LB 2.99
CUCUMBERS 1.49 EACH 1.49
BELL PEPPERS 3.99 LB @ 3.99/LB 3.99

DAIRY DEPARTMENT
WHOLE MILK 3.99 GALLON 3.99
GREEK YOGURT 1.99 6PK 1.99
CHEDDAR CHEESE 5.99 LB 5.99
BUTTER 4.99 LB 4.99

BAKERY DEPARTMENT
WHOLE WHEAT BREAD 2.99 LOAF 2.99
CROISSANTS 3.99 6PK 3.99
CHOCOLATE CAKE 12.99 EACH 12.99

FROZEN FOODS
FROZEN PEAS 1.99 BAG 1.99
FROZEN CORN 1.99 BAG 1.99
ICE CREAM 5.99 QT 5.99

BEVERAGES
ORANGE JUICE 4.99 64OZ 4.99
SPARKLING WATER 2.99 6PK 2.99

SUBTOTAL 67.63
TAX 5.41
TOTAL 73.04

PAYMENT: VISA ****1234
AUTH: 987654
DATE/TIME: 03/15/2024 14:32:15

THANK YOU FOR SHOPPING AT WALMART
SAVE MONEY. LIVE BETTER.
CUSTOMER SERVICE: 1-800-WALMART"""
        ]

        for i, text in enumerate(good_texts):
            with self.subTest(good_text=f"good_text_{i+1}"):
                # Test simple scoring
                score = self.ocr_service.score_ocr_quality(text)
                self.assertGreaterEqual(score, 0.5, f"Good text {i+1} should score >= 0.5, got {score}")
                self.assertLessEqual(score, 1.0, f"Score should be <= 1.0, got {score}")

                # Test detailed scoring
                detailed = self.ocr_service.score_ocr_quality(text, detailed=True)
                self.assertEqual(detailed['total_score'], score)
                self.assertGreaterEqual(detailed['component_scores']['text_length'], 0.4)
                self.assertGreaterEqual(detailed['component_scores']['price_patterns'], 0.6)
                self.assertGreaterEqual(detailed['component_scores']['total_keyword'], 0.6)
                self.assertGreaterEqual(detailed['component_scores']['word_quality'], 0.8)
                self.assertLessEqual(detailed['component_scores']['noise_penalty'], 0.3)

    def test_bad_ocr_text_low_score(self):
        """Test that bad OCR text receives low scores"""
        bad_texts = [
            # Empty text
            "",

            # Very short text
            "TOTAL",
            "5.99",
            "CASH",

            # No structure, just noise
            "@@@ ### !!! 123 456 789 *** &&& %%% ###",
            "asdfghjklqwertyuiopzxcvbnm",
            "1234567890!@#$%^&*()_+-=[]{}|;:,.<>?",

            # Text without receipt features
            """This is just some random text that doesn't look like a receipt at all.
It doesn't have any prices or totals or anything that would indicate it's
a receipt from a store. It's just regular text with sentences and paragraphs
but no monetary values or receipt-specific terminology.""",

            # Only numbers, no context
            """123 456 789
101 202 303
404 505 606
707 808 909""",

            # Only special characters
            """!@#$%^&*()
_+-=[]{}|;:',.<>/?""",

            # Mixed but no receipt structure
            """Hello World
This is a test
123 Main Street
Anytown, USA 12345
(555) 123-4567
email@example.com
http://www.example.com""",

            # Single characters repeated
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "111111111111111111111111111111",
            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",

            # Very short with no meaningful content
            "x",
            "1",
            "$",
            ".",

            # Gibberish with some numbers
            "asdf 123 qwer 456 zxcv 789 !@# 123 $%^ 456 &*() 789"
        ]

        for i, text in enumerate(bad_texts):
            with self.subTest(bad_text=f"bad_text_{i+1}"):
                # Test simple scoring
                score = self.ocr_service.score_ocr_quality(text)
                self.assertLessEqual(score, 0.4, f"Bad text {i+1} should score <= 0.4, got {score}")
                self.assertGreaterEqual(score, 0.0, f"Score should be >= 0.0, got {score}")

                # Test detailed scoring
                detailed = self.ocr_service.score_ocr_quality(text, detailed=True)
                self.assertEqual(detailed['total_score'], score)

                # Check specific component failures
                if len(text.strip()) < 50:
                    self.assertLess(detailed['component_scores']['text_length'], 0.3)
                if not any(price in text.lower() for price in ['$', '3.99', '5.99', '12.99']):
                    self.assertLess(detailed['component_scores']['price_patterns'], 0.3)
                if 'total' not in text.lower() and 'amount' not in text.lower() and 'balance' not in text.lower():
                    self.assertLess(detailed['component_scores']['total_keyword'], 0.3)

    def test_medium_ocr_text_medium_score(self):
        """Test that medium quality OCR text receives medium scores"""
        medium_texts = [
            # Short but complete receipt
            """STORE
TOTAL 5.99
CASH 5.99""",

            # Receipt with some structure but missing elements
            """GROCERY
APPLES 3.99
ORANGES 2.50
BANANAS 1.25
CASH 7.74""",

            # Receipt with prices but no total
            """GROCERY STORE
MILK 4.99
BREAD 2.99
EGGS 3.99
CHEESE 5.99""",

            # Receipt with total but few prices
            """SUPERMARKET
TOTAL 25.99
TAX 2.08
PAYMENT: CREDIT CARD""",

            # Receipt with formatting issues
            """GROCERY STORE RECEIPT
APPLES 3 99
ORANGES 2 50
BANANAS 1 25
TOTAL 7 74
CASH 7 74""",

            # Medium length with some noise
            """STORE RECEIPT
ITEM1 4.99
ITEM2 2.50
@@@ ###
TOTAL 7.49
THANKS""",

            # Receipt with missing components
            """GROCERY
MILK 4.99
BREAD 2.99
SUBTOTAL 7.98""",

            # Receipt with some structure but short
            """TOTAL AMOUNT: $12.99
TAX: $1.04
CASH: $14.03"""
        ]

        for i, text in enumerate(medium_texts):
            with self.subTest(medium_text=f"medium_text_{i+1}"):
                # Test simple scoring
                score = self.ocr_service.score_ocr_quality(text)
                self.assertGreaterEqual(score, 0.3, f"Medium text {i+1} should score >= 0.3, got {score}")
                self.assertLessEqual(score, 0.7, f"Medium text {i+1} should score <= 0.7, got {score}")

                # Test detailed scoring
                detailed = self.ocr_service.score_ocr_quality(text, detailed=True)
                self.assertEqual(detailed['total_score'], score)

    def test_component_scoring_logic(self):
        """Test specific component scoring logic"""
        # Test text length scoring
        short_text = "TOTAL 5.99"
        medium_text = """GROCERY STORE
APPLES 3.99
ORANGES 2.50
TOTAL 6.49"""
        long_text = """GROCERY STORE RECEIPT
APPLES 3.99
ORANGES 2.50
BANANAS 1.25
MILK 4.99
BREAD 2.99
CHEESE 5.99
YOGURT 1.99
CEREAL 3.99
COFFEE 7.99
TEA 3.49
JUICE 4.99
CHIPS 3.99
COOKIES 2.99
CANDY 1.99
NUTS 4.99
SODA 5.99
WATER 3.99
SOAP 3.99
SHAMPOO 5.99
PAPER TOWELS 8.99
TRASH BAGS 4.99
SUBTOTAL 85.67
TAX 6.85
TOTAL 92.52
CASH 92.52
THANK YOU"""

        short_detailed = self.ocr_service.score_ocr_quality(short_text, detailed=True)
        medium_detailed = self.ocr_service.score_ocr_quality(medium_text, detailed=True)
        long_detailed = self.ocr_service.score_ocr_quality(long_text, detailed=True)

        # Text length should increase with text length (but may be capped at max)
        self.assertLessEqual(short_detailed['component_scores']['text_length'],
                           medium_detailed['component_scores']['text_length'])
        self.assertLessEqual(medium_detailed['component_scores']['text_length'],
                           long_detailed['component_scores']['text_length'])

        # Test price pattern scoring
        no_prices = "GROCERY STORE\nAPPLES\nORANGES\nTOTAL"
        some_prices = "GROCERY STORE\nAPPLES 3.99\nORANGES\nTOTAL"
        many_prices = "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nBANANAS 1.25\nMILK 4.99\nTOTAL"

        no_prices_detailed = self.ocr_service.score_ocr_quality(no_prices, detailed=True)
        some_prices_detailed = self.ocr_service.score_ocr_quality(some_prices, detailed=True)
        many_prices_detailed = self.ocr_service.score_ocr_quality(many_prices, detailed=True)

        # Price pattern score should increase with more prices
        self.assertLess(no_prices_detailed['component_scores']['price_patterns'],
                        some_prices_detailed['component_scores']['price_patterns'])
        self.assertLess(some_prices_detailed['component_scores']['price_patterns'],
                        many_prices_detailed['component_scores']['price_patterns'])

        # Test total keyword scoring
        no_total = "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nCASH 6.49"
        with_total = "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49"
        with_amount = "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nAMOUNT 6.49\nCASH 6.49"
        with_balance = "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nBALANCE 6.49\nCASH 6.49"

        no_total_detailed = self.ocr_service.score_ocr_quality(no_total, detailed=True)
        with_total_detailed = self.ocr_service.score_ocr_quality(with_total, detailed=True)
        with_amount_detailed = self.ocr_service.score_ocr_quality(with_amount, detailed=True)
        with_balance_detailed = self.ocr_service.score_ocr_quality(with_balance, detailed=True)

        # Total keyword score should be higher with keywords
        self.assertLessEqual(no_total_detailed['component_scores']['total_keyword'],
                            with_total_detailed['component_scores']['total_keyword'])
        self.assertLessEqual(no_total_detailed['component_scores']['total_keyword'],
                            with_amount_detailed['component_scores']['total_keyword'])
        self.assertLessEqual(no_total_detailed['component_scores']['total_keyword'],
                            with_balance_detailed['component_scores']['total_keyword'])

        # Test noise penalty scoring
        clean_text = "GROCERY STORE\nAPPLES 3.99\nTOTAL 6.49"
        noisy_text = "@@@ ### !!! GROCERY STORE ### @@@\nAPPLES 3.99 *** &&&\nTOTAL 6.49 %%% $$$"
        very_noisy = "@@@ ### !!! *** &&& %%% $$$ @@@ ### !!! *** &&& %%% $$$"

        clean_detailed = self.ocr_service.score_ocr_quality(clean_text, detailed=True)
        noisy_detailed = self.ocr_service.score_ocr_quality(noisy_text, detailed=True)
        very_noisy_detailed = self.ocr_service.score_ocr_quality(very_noisy, detailed=True)

        # Noise penalty should increase with more noise (but may be capped at max)
        self.assertLessEqual(clean_detailed['component_scores']['noise_penalty'],
                           noisy_detailed['component_scores']['noise_penalty'])
        self.assertLessEqual(noisy_detailed['component_scores']['noise_penalty'],
                           very_noisy_detailed['component_scores']['noise_penalty'])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test None input - should handle gracefully
        try:
            score = self.ocr_service.score_ocr_quality(None)
            # If it doesn't raise an exception, check the score is reasonable
            self.assertEqual(score, 0.0)
        except (AttributeError, TypeError):
            # If it raises an exception, that's also acceptable
            pass

        # Test whitespace only
        whitespace_only = "   \n\t   \n   \t   "
        score = self.ocr_service.score_ocr_quality(whitespace_only)
        self.assertEqual(score, 0.0)

        detailed = self.ocr_service.score_ocr_quality(whitespace_only, detailed=True)
        self.assertEqual(detailed['total_score'], 0.0)
        # text_stats may not be present for empty text
        if 'text_stats' in detailed:
            self.assertEqual(detailed['text_stats']['length'], 0)
            self.assertEqual(detailed['text_stats']['word_count'], 0)

        # Test very long text
        very_long_text = "WORD " * 10000  # Very long repetitive text
        score = self.ocr_service.score_ocr_quality(very_long_text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test Unicode characters
        unicode_text = """CAFÉ RECEIPT
CRÈME 4.99
CAFÉ AU LAIT 3.99
NAÏVE 2.99
RÉSUMÉ 1.99
TOTAL 13.96"""
        score = self.ocr_service.score_ocr_quality(unicode_text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test mixed case
        mixed_case = """gRoCeRy StOrE
ApPlEs 3.99
OrAnGeS 2.50
tOtAl 6.49
CaSh 6.49"""
        score = self.ocr_service.score_ocr_quality(mixed_case)
        self.assertGreaterEqual(score, 0.3)  # Should recognize despite case

    def test_scoring_consistency(self):
        """Test that scoring is consistent and deterministic"""
        text = """GROCERY STORE
APPLES 3.99
ORANGES 2.50
TOTAL 6.49
CASH 6.49"""

        # Multiple calls should return same result
        score1 = self.ocr_service.score_ocr_quality(text)
        score2 = self.ocr_service.score_ocr_quality(text)
        score3 = self.ocr_service.score_ocr_quality(text)

        self.assertEqual(score1, score2)
        self.assertEqual(score2, score3)

        # Detailed scoring should be consistent
        detailed1 = self.ocr_service.score_ocr_quality(text, detailed=True)
        detailed2 = self.ocr_service.score_ocr_quality(text, detailed=True)

        self.assertEqual(detailed1['total_score'], detailed2['total_score'])
        self.assertEqual(detailed1['component_scores'], detailed2['component_scores'])
        self.assertEqual(detailed1['raw_scores'], detailed2['raw_scores'])

    def test_scoring_ranges(self):
        """Test that all scoring components are within expected ranges"""
        text = """GROCERY STORE RECEIPT
APPLES 3.99
ORANGES 2.50
BANANAS 1.25
MILK 4.99
BREAD 2.99
TOTAL 15.72
TAX 1.26
TOTAL AMOUNT 16.98
CASH 16.98
THANK YOU"""

        detailed = self.ocr_service.score_ocr_quality(text, detailed=True)

        # Check total score range
        self.assertGreaterEqual(detailed['total_score'], 0.0)
        self.assertLessEqual(detailed['total_score'], 1.0)

        # Check component score ranges
        for component, score in detailed['component_scores'].items():
            self.assertGreaterEqual(score, 0.0, f"Component {component} should be >= 0.0")
            self.assertLessEqual(score, 1.0, f"Component {component} should be <= 1.0")

        # Check raw score ranges
        for component, score in detailed['raw_scores'].items():
            if component == 'noise_penalty_points':
                self.assertGreaterEqual(score, 0.0, f"Raw {component} should be >= 0.0")
                self.assertLessEqual(score, 10.0, f"Raw {component} should be <= 10.0")
            else:
                self.assertGreaterEqual(score, 0.0, f"Raw {component} should be >= 0.0")
                # Check against max scores
                max_score = detailed['max_scores'][component.replace('_points', '')]
                self.assertLessEqual(score, max_score, f"Raw {component} should be <= {max_score}")


if __name__ == '__main__':
    unittest.main()
