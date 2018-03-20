


def split_lines(text):
    return text.split("//")


def is_constonant(char):

    constonants = "bcdfghjklmnpqrstvxz"
    vowels = "aeiouy"

    return char.lower() in constonants


def is_haiku(text):

    lines = split_lines(text)
    syllables_per_line = [count_line_syllables(line) for line in lines]
    print(syllables_per_line)

    return syllables_per_line == [5, 7, 5]


def count_line_syllables(line):
    words = line.split()
    print(words)
    count = sum([count_word_syllables(w) for w in words])
    return count


def count_word_syllables(word):

    count = 1

    current_is_constonant = True
    char_is_constonant = True

    if not is_constonant(word[0]) and any(is_constonant(x) for x in word):
        count -= 1

    for char in word:
        char_is_constonant = is_constonant(char)

        if char_is_constonant:
            if not current_is_constonant:
                count += 1

        current_is_constonant = char_is_constonant

    return count

# def count_word_syllables(word):
#
#     count = 0
#
#     current_is_constonant = False
#     char_is_constonant = False
#
#     for char in word:
#         char_is_constonant = is_constonant(char)
#
#         if not char_is_constonant:
#             if current_is_constonant:
#                 count += 1
#
#         current_is_constonant = char_is_constonant
#
#     return count


def main():
    return

if __name__ == "__main__":
    main()