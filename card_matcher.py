from jaro import jaro_winkler_metric
import json
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)


class CardMatcher():
    """
    CardMatcher takes as input a JSON file that is a dictionary of card set 
    abbreviations, where each of those card sets then contain a dictionary
    containing potential card names that are mapped to the standardized card
    name. This can be done for any cards as needed, but was initially done to
    take into account the formatting for Split Cards and needing a single
    card name (e.g. "Consign // Oblivion"). This is because the model identifies 
    the "Card Name" section of the card and Split Cards would be classified as 
    two separate cards. 

    Once the OCR text of the card is extracted, the CardMatcher maps the text
    to the best Jaro-Winkler score among the specified card sets. The mapped
    card name is then returned to the user.
    """
    def __init__(self, path:str=os.path.join(__location__, "card_list.json")):
        with open(path) as f:
            self.card_list = json.load(f)
        self.card_sets = list(self.card_list.keys())

    def select_sets(self, card_subsets:list=None) -> None:
        """
        Select the card sets that we will consider when mapping the extracted 
        OCR card text. Defaults to looking at all card sets in the provided file. 

        Parameters
        ----------
        card_subsets : List(string)
            Should contain only valid MTG card set abbreviations (e.g. "MKM", "LCI")
        """
        # If no card subsets are specified, default to all of the keys (i.e. card sets) in the loaded file
        if not card_subsets:
            card_subsets = self.card_sets

        # Combine the card dictionaries into a single dictionary
        card_selection = {}
        for cs in card_subsets:
            # Only merge on valid dictionary keys
            if self.card_list.get(cs):
                card_selection = card_selection | self.card_list[cs]

        self.cards = card_selection

    def match(self, card_string:str) -> str:
        """
        Find the standardized card name with the highest Jaro-Winkler
        score relative to the incoming card name. 

        Parameters
        ----------
        card_string : string
            The string extracted from the card we're attempting to identify.

        Returns
        -------
        mapped_name : string
            The standardized name for the card with the best matched card name. 
        """
        card_scores = {}
        score_threshold = 0.50
        # Compute similarity score with each card
        for card in self.cards:
             score = jaro_winkler_metric(card_string, card)
             card_scores[card] = score

        # Find the card with the highest score
        v = list(card_scores.values())
        k = list(card_scores.keys())
        max_score = max(v)
        best_match = k[v.index(max_score)]
        # Only want to consider GOOD matches; return a blank string otherwise
        mapped_name = self.cards[best_match] if max_score > score_threshold else ""

        return mapped_name