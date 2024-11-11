

import random
import argparse
import codecs
import os
import numpy
from jinja2.compiler import generate
from sympy import sequence


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        # Fill out transitions dictionary
        file = open(basename + '.trans')
        d1 = dict()
        d2 = dict()
        start_word = 'placeholder'
        for line in file :
            i = 0
            l = line.split()
            if l[0] != start_word :
                if start_word != 'placeholder' :
                    d1[start_word] = d2
                    d2 = dict()
                start_word = l[0]
            d2[l[1]] = l[2]
        d1[start_word] = d2
        self.transitions = d1

        # Fill out emissions dictionary
        file = open(basename + '.emit')
        d1 = dict()
        d2 = dict()
        start_word = 'placeholder'
        for line in file :
            i = 0
            l = line.split()
            if l[0] != start_word :
                if start_word != 'placeholder' :
                    d1[start_word] = d2
                    d2 = dict()
                start_word = l[0]
            d2[l[1]] = l[2]
        d1[start_word] = d2
        self.emissions = d1


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        # pass
        trans = ''
        emiss = ''
        state = '#'
        i = 0

        # make transmisons states seq
        while i < n :
            k = self.transitions[state].keys()
            v = self.transitions[state].values()
            v2 = []
            k2 = []
            for val in v :
                val = float(val) * 10000
                v2.append(int(val))
            for item in k :
                item = str(item)
                k2.append(item)
            choice = random.choices(k2, weights=v2)
            state = choice[0]
            trans = trans + (state + ' ')
            i = i + 1

        # make emissions states seq
        states = trans.split()
        for value in states :
            k = self.emissions[value].keys()
            v = self.emissions[value].values()
            v2 = []
            k2 = []
            for val in v:
                val = float(val) * 10000
                v2.append(int(val))
            for item in k:
                item = str(item)
                k2.append(item)
            choice = random.choices(k2, weights=v2)
            state = choice[0]
            emiss = emiss + (state + ' ')

        return Sequence(trans, emiss)



    def forward(self, sequence):
    ## you do this: Implement the Forward algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

        # set up the matrix
        O = sequence.outputseq.split()
        col = len(O) #TODO might need to change back to + 1
        rows = len(self.emissions.keys())
        M = numpy.zeros((rows,col))

        ## set up the initial probabilities from the start state (states[0] to observation 1.
        ## T is the transition probabilities, E is the emission probabilities
        ## O is the vector of observations.

        m = 0 # this will be placeholder for rows [happy, grumpy, hungry]
        for s in self.emissions.keys():
            sub_trans = self.transitions[s]
            T = sub_trans[s]
            sub_emiss = self.emissions[s]
            E = sub_emiss[O[0]]
            prob = float(T) * float(E)
            M[m, 0] = prob  ## the probability of that state * the probability of that state given observation 1.
            #TODO ^ may need to change back to [i,1]
            m = m + 1

        t = col
        for i in range(1, t): #TODO may need to change back to (2,t)
            m = -1
            for s in self.emissions.keys():
                sum = 0
                m = m + 1# this will be placeholder for rows [happy, grumpy, hungry]
                for s2 in self.emissions.keys():
                    sub_trans = self.transitions[s2]
                    T = sub_trans[s]
                    thing = O[i]
                    sub_emiss = self.emissions[s2]
                    E = sub_emiss[O[i]]
                    sum += M[m, i - 1] * float(T) * float(E)
                M[m, i] = sum

        max = 0
        i = 0
        m = 0 # this will be placeholder for rows [happy, grumpy, hungry]
        best = 0
        while i < rows :
            if M[i,col-1] > max :
                max = M[i,col-1]
                best = m
            i = i + 1
            m = m + 1
        states = list(self.emissions.keys())
        state = states[best]
        return state



    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.


if __name__ == "__main__" :
    h = HMM()
    h.load('cat')
    # h.load('partofspeech')
    # print(h.transitions)
    # print(h.emissions)
    l = h.generate(5)
    k = h.forward(l)
    print(k)

