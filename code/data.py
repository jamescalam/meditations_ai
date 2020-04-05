"""
Script to control import of Stoic philosophical works in text format.

Currently imports:
    - The Meditations, Marcus Aurelius
    - Ad Lucilium Epistulae Morales, Seneca
"""

import requests
import re
from bs4 import BeautifulSoup


def meditations():
    """
    Imports the meditations by Marcus Aurelius.
    :return meditations_txt:
    """
    # import Meditations by Marcus Aurelius
    response = requests.get('http://classics.mit.edu/Antoninus/meditations.mb.txt')
    data = response.text
    del response

    # remove everything before and including "Translated by George Long"
    data = data.split('Translated by George Long')[1]

    # remove "----" lines
    data = re.sub(r'([-])\1+', '', data)

    # remove "BOOK ..." lines, for this we use regular expressions
    data = re.sub('BOOK [A-Z]+\n', '', data)

    # remove "THE END" and all that follows it
    data = data.split("THE END")[0]

    # splitting by newline characters
    data = data.split('\n\n')

    # remove empty samples
    empty = lambda x: x.replace('\s+', '') != ''
    data = list(filter(empty, data))

    # remove final '\n' characters
    data = list(map(lambda x: x.replace('\n', ' '), data))

    print(f"We have {len(data)} stoic lessons from Marcus Aurelius")

    # now join back together in full text
    meditations_txt = '\n'.join(map(lambda x: x.strip(), data))  # we also use map to strip each paragraph

    return meditations_txt


def hello_lucilius(jupyter=False):
    """
    Imports 'Ad Lucilium Epistulae Morales' by Seneca
    :return:
    """

    # import page containing links to all of Seneca's letters
    # get web address
    src = "https://en.wikisource.org/wiki/Moral_letters_to_Lucilius"

    html = requests.get(src).text  # pull html as text
    soup = BeautifulSoup(html, "html.parser")  # parse into BeautifulSoup object

    # create function to pull letter from webpage (pulls text within <p> elements
    def pull_letter(http):
        print(f"Pulling {http.split('/')[-1:][0]}")
        # get html from webpage given by 'http'
        html = requests.get(http).text
        # parse into a beautiful soup object
        soup = BeautifulSoup(html, "html.parser")

        # build text contents within all p elements
        txt = '\n'.join([x.text for x in soup.find_all('p')])
        # replace extended whitespace with single space
        txt = txt.replace('  ', ' ')
        # replace webpage references ('[1]', '[2]', etc)
        txt = re.sub('\[[0-9]+\]', '', txt)
        # replace all number bullet points that Seneca uses ('1.', '2.', etc)
        txt = re.sub('[0-9]+. ', '', txt)
        # remove double newlines
        return txt.replace("\n\n", "\n")

    # compile RegEx for finding 'Letter 12', 'Letter 104' etc
    letters_regex = re.compile("^Letter\s+[0-9]{1,3}$")
    # create dictionary containing letter number: [local href, letter contents] for all that satisfy above RegEx
    return {x.contents[0]: [x.get('href'), pull_letter(f"https://en.wikisource.org{x.get('href')}")]
            for x in soup.find_all('a')
            if len(x.contents) > 0
            if letters_regex.match(str(x.contents[0]))}
