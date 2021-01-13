import pandas as pd
import re
from string import punctuation
import requests
import xml.etree.ElementTree as ET
from collections import Counter


def create_file(url, name):
    """
    Creates an xml file from the url given.

    :param url: URL at which the xml file is located
    :param name: name of the desired xml file to create
    :return: None
    """
    cntnt = requests.get(url).content
    txtfile = open(name, "w", encoding="utf-8")
    txtfile.write(cntnt.decode(encoding="utf-8"))


def parse_xml_file(file):
    """
    Parses an xml file and creates a dataframe with all labeled verses.

    :param file: An xml file
    :return: A dataframe. First column is the verse id, and second column is the verse text.
    """
    verse_list = []
    bible_tree = ET.parse(file)
    root = bible_tree.getroot()

    for element in root.iter():
        if element.get("type") == "verse":
            verse_list.append([element.get("id"), element.text.strip()])

    bible_df = pd.DataFrame(verse_list, columns=["Verse", "Text"])
    return bible_df


def clean_text(text_list):
    """
    Removes certain special characters, puts spaces between all punctuation and words, and makes it lowercase.

    :param text_list: list of strings (verses)
    :return: modifies list in-place
    """
    for i in range(len(text_list)):
        # Get rid of inverted punctuation and such
        text_list[i] = re.sub('([' + chr(161) + chr(191) + '])', '', text_list[i])

        # Put spaces between all words and punctuation, (except apostrophes since those are part of the word)
        text_list[i] = re.sub('([' + punctuation.replace("\'", "") + '])', r' \1 ', text_list[i])

        # Get rid of all multiple spaces in a row
        text_list[i] = re.sub(' +', ' ', text_list[i])

        # Make it lowercase
        text_list[i] = text_list[i].lower()


# Write to xml files from online URLs
create_file("https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Spanish.xml", "Spanish.xml")
create_file("https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/English.xml", "English.xml")

# Parse the xml's into useful dataframes
spanish_df = parse_xml_file("Spanish.xml")
english_df = parse_xml_file("English.xml")

# Clean the bible text
clean_text(english_df["Text"])
clean_text(spanish_df["Text"])

# Get list of words in order
english_word_list = ("".join(english_df["Text"])).split(" ")
english_word_list = [x for x in english_word_list if x != ""]

spanish_word_list = (" . ".join(spanish_df["Text"])).split(" ")
spanish_word_list = [x for x in spanish_word_list if x != ""]

# Get vocabulary sets
english_vocab_set = set(english_word_list)
spanish_vocab_set = set(spanish_word_list)

# Create counter objects
english_counter = Counter(english_word_list)
spanish_counter = Counter(spanish_word_list)







