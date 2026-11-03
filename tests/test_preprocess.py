from spamdetector.preprocess import preprocess


def test_lowercases_and_strips_punctuation():
    assert preprocess("Hello, WORLD!!") == "hello world"


def test_urls_and_numbers_are_replaced_with_tokens():
    assert preprocess("Visit www.x.com or http://a.b/c now") == "visit url or url now"
    assert preprocess("Call 0800 55055 now") == "call num num now"


def test_whitespace_is_collapsed():
    assert preprocess("  a \n\t b  ") == "a b"


def test_matches_original_notebook_behaviour():
    # Worked example from the SMS Spam Collection
    assert preprocess("Text FA to 87121 to receive entry question(std txt rate)T&C's apply") == \
        "text fa to num to receive entry question std txt rate t c s apply"
