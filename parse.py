#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
from collections import defaultdict
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        
        self.tokens = tokens
        self.grammar = grammar

        self.R = defaultdict(list)
        self.P = defaultdict(set)
        for non_terminal, rules in grammar._expansions.items():
            for rule in rules:
        # Now, you can

                lhs, rhs = rule.lhs, rule.rhs
                if rhs:
                    self.R[rhs[0]].append(rule)
                    self.P[rhs[0]].add(lhs)
        
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def _compute_Sj(self, j: int) -> Set[str]: #! this is the part I edited
        """Compute the S_j table for position j."""
        Sj = set()
        if j >= len(self.tokens):
            return Sj #meaning that we are out of bound
        wj = self.tokens[j]
        agenda = [wj]
        while agenda: #essentially adding all ancestors of this word to Sj
            B = agenda.pop()
            for A in self.P.get(B, []):
                if A not in Sj:
                    Sj.add(A)
                    agenda.append(A)
        return Sj

    def acceptedBP(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        minWeight = -1
        curResult = ""
        for item in self.cols[-1].all():    # the last column
            if item is None:
                continue
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0

                    #! goal here is to do backward and get the correct parse:
                    stack = [] #use a stack to throw in the elements
                    stack.append((item,0))
                    result = "" #the result string that is printed out

                    depthCount = 0

                    tempAdjust = 0 #temporary adjustment used to handle the case of terminals(basically it doesn't have parenthesis after it)
                    while stack:
                        toTerminal = False
                        i,curDepth = stack.pop()

                       # print(i)
                       # print("i depth:",curDepth)
                       # print("curdapth:",depthCount)


                        if isinstance(i, str):
                            numParen = depthCount - curDepth + 1 + tempAdjust if depthCount - curDepth + 1 > 0 and depthCount != 0 else 0
                            result += ")"*numParen
                            result += i+" "
                            depthCount = curDepth
                            tempAdjust = -1
                            continue
                        #if(i.dot_position == 0):
                            #depthCount += 1
                        if(i.dot_position == len(i.rule.rhs)):
                            numParen = depthCount - curDepth + 1 + tempAdjust if depthCount - curDepth + 1 > 0 and depthCount != 0 else 0
                            result += ")"*numParen
                            result += ("("+i.rule.lhs+" ")
                            if i.dot_position == 1 and not self.grammar.is_nonterminal(i.rule.rhs[0]):
                                result +=  (i.rule.rhs[0])
                                toTerminal = True
                            depthCount = curDepth
                        
                        #print(i.position)
                        #print(i)
                        #print(self.cols[i.position]._back)
                        tempAdjust = 0
                        if toTerminal:
                            continue

                        temp = self.cols[i.position]._back[i]
                        if len(temp) == 1: #this must be a scan
                            stack.append((i.rule.rhs[i.dot_position-1],depthCount+1))
                            stack.append((temp[0],depthCount+1))
                        elif len(temp) == 2: #this is the attach case
                           # print("did it with depthCount",depthCount+1)
                            stack.append((temp[1],depthCount+1))
                            stack.append((temp[0],depthCount+1))
                            
                    result += (")"*(depthCount+1+tempAdjust))

                    if minWeight == -1 or minWeight > self.cols[-1]._weight[item]:
                        curResult = result
                        minWeight = self.cols[-1]._weight[item]
        
        if minWeight == -1:
            print("NONE")  # we didn't find any appropriate item
        else:
            print(curResult)
            print(minWeight)

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            Sj = self._compute_Sj(i)
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    if next not in Sj: #if our nonterminal isn't one of these ancestors, we can safely ignore it
                        continue
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)    
                #print("current column:",column)                  

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        #print("predicted"position)
        
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position, position = position)

            # modified part
            if new_item not in self.cols[position]._index:
                if self.grammar.is_nonterminal(new_item.next_symbol()):
                    self.cols[position].push(new_item,(),(),new_item.rule.weight) #!modified here(changed push so that it includes the backward tuple and the weight)
                elif new_item.next_symbol() in self.tokens and position >= self.tokens.index(new_item.next_symbol()):
                    self.cols[position].push(new_item,(),(),new_item.rule.weight)
            # ended modification

            log.debug(f"\tPredicted: {new_item} in column {position}")
            
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        #print(position)
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            # print(f"{position}")
            # print(item.next_symbol())
            new_item = item.with_dot_advanced(1)
            #print(f"\tScanned to get: {new_item} in column {position+1}")
            self.cols[position + 1].push(new_item,(item,),(self.cols[position]._weight[item],),0) #!modified here(0 here stand for the weight of new item other than the sum of its children)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            
            self.profile["SCAN"] += 1


    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left



        for customer in self.cols[mid].expectations.get(item.rule.lhs, []):
            new_item = customer.with_dot_advanced(item.position-customer.position)
            self.cols[position].push(new_item, (customer, item), (self.cols[mid]._weight[customer], self.cols[position]._weight[item]), 0)
            log.debug(f"\tAttached to get: {new_item} in column {position}")
            self.profile["ATTACH"] += 1



class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.


    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._back = {} #! Dictionary used to store the back pointer(aka two other Items)
        self._weight: Dict[Item, float] = {} #! save the weight of each item
        self._next = 0                     # index of first item that has not yet been popped
        self.expectations = defaultdict(list)


        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item, back: Tuple(),backWeights: Tuple(),extraWeight: float) -> None:
        #print("")
        curWeight = extraWeight
        for w in backWeights:
            curWeight += w

        #print("newWeight:",curWeight)
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:    # O(1) lookup in hash table
            #print("item:",item)
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            self._weight[item] = curWeight
            self._back[item] = back
            #print("with weight",curWeight)
            next_symbol = item.next_symbol()
            self.expectations[next_symbol].append(item) #categorize item by the next symbol
        else: #! if the item is in the hashtable, we see if the weight can be improved
            #if len(backWeights) == 0:
            #    return #if we are predicting something that we already have, just return it
            #if len(backWeights) == 1:
                #print("WHAT!") #this shouldn't happen
            #    return

            #print(curWeight)
            #print(self._weight[item])
            if curWeight  < self._weight[item]:
                #print("update with from",self._weight[item],"to",curWeight)
                self._weight[item] = curWeight
                self._back[item] = back
                
                next_symbol = item.next_symbol()

                self.expectations[next_symbol].remove(self._items[self._index[item]]) #kill it in expectations
                self._items[self._index[item]] = None #kill the useless thing itself
                self._items.append(item)
                self._index[item] = len(self._items) - 1
                self.expectations[next_symbol].append(item)
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = None
        while item == None:
            item = self._items[self._next]
            self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    #! Seems to work pretty well without the need of any modifications
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    #! Seems to work pretty well without the need of any modifications
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self,advancement) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position,position = self.position+advancement)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                #! print the result --> acceptedBP includes printing method within
                chart.acceptedBP()
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
