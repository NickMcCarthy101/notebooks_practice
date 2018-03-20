import pytest
from haiku_checker import split_lines, count_word_syllables, count_line_syllables, is_haiku



@pytest.fixture()
def fake_poem():
    poem = "Is this a haiku//Who knows//Hmmmmmm"
    return poem

@pytest.fixture()
def real_poem():
    poem = "Is this a haiku//Who knows if it is true huh//The Pineapple knows"
    return poem


class TestHaiku():

    def test_split_lines(self, fake_poem):


        lines = split_lines(fake_poem)

        assert len(lines) == fake_poem.count('//') + 1
        assert lines[0] == 'Is this a haiku'
        assert lines[1] == 'Who knows'
        assert lines[2] == 'Hmmmmmm'

    def test_count_word_syllables(self):
        assert count_word_syllables('Hello') == 2
        assert count_word_syllables('He') == 1
        assert count_word_syllables('These') == 2
        assert count_word_syllables('Howdy') == 2
        assert count_word_syllables('Pineapple') == 3
        assert count_word_syllables('Is') == 1
        assert count_word_syllables('A') == 1
        assert count_word_syllables('haiku') == 2
        assert count_word_syllables('who') == 1
        assert count_word_syllables('knows') == 1

    def test_count_line_syllables(self):

        line = "Hello he these howdy pineapple"
        assert count_line_syllables(line) == 10


    def test_is_haiku(self, fake_poem, real_poem):

        answer = is_haiku(fake_poem)

        assert answer is False

        answer = is_haiku(real_poem)

        assert answer is True