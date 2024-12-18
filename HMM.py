

import random
import argparse
import codecs
import os
import sys

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
        file.close()

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
        file.close()


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
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
        col = len(O) + 1
        rows = len(self.transitions.keys())
        M = numpy.zeros((rows,col))

        ## set up the initial probabilities from the start state (states[0] to observation 1.
        M[0,0] = 1.0
        ## T is the transition probabilities, E is the emission probabilities
        ## O is the vector of observations.

        m = 1 # this will be placeholder for rows [happy, grumpy, hungry]
        for s in self.emissions.keys():
            sub_trans = self.transitions['#']
            if s in sub_trans :
                T = sub_trans[s]
            else:
                T = 0
            sub_emiss = self.emissions[s]
            if O[0] in sub_emiss :
                E = sub_emiss[O[0]]
            else :
                E = 0
            prob = float(T) * float(E)
            M[m, 1] = prob  ## the probability of that state * the probability of that state given observation 1.
            m = m + 1

        t = col
        for i in range(2, t):
            m = 0
            for s in self.emissions.keys():
                sum = 0
                m = m + 1# this will be placeholder for rows [happy, grumpy, hungry]
                m2 = 1# this will be placeholder for rows [happy, grumpy, hungry]
                E = 0.0
                for s2 in self.emissions.keys():
                    if m2 == 1 :
                        sub_emiss = self.emissions[s]
                        if O[i-1] in sub_emiss:
                            E = sub_emiss[O[i-1]]
                        else :
                            E = 0
                    sub_trans = self.transitions[s2]
                    if s in sub_trans:
                        T = sub_trans[s]
                    else:
                        T = 0
                    thing = O[i-1]
                    prev = M[m2, i - 1]
                    sum += prev * float(T) * float(E)
                    m2 = m2 + 1
                M[m, i] = sum

        # returns the state corresponding with the highest value in the last column of matrix
        max = 0
        i = 1
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
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.
        # set up the matrix
        O = sequence.outputseq.split()
        col = len(O) + 1
        rows = len(self.transitions.keys())
        N = numpy.zeros((rows,col))
        bp = numpy.zeros((rows,col))

        ## set up the initial probabilities from the start state (states[0] to observation 1.
        N[0,0] = 1.0
        ## T is the transition probabilities, E is the emission probabilities
        ## O is the vector of observations.

        m = 1 # this will be placeholder for rows [happy, grumpy, hungry]
        for s in self.emissions.keys():
            sub_trans = self.transitions['#']
            if s in sub_trans :
                T = sub_trans[s]
            else:
                T = 0
            sub_emiss = self.emissions[s]
            if O[0] in sub_emiss:
                E = sub_emiss[O[0]]
            else:
                E = 0
            prob = float(T) * float(E)
            N[m, 1] = prob  ## the probability of that state * the probability of that state given observation 1.
            m = m + 1

        t = col
        for i in range(2, t):
            m = 0
            for s in self.emissions.keys():
                prod_list = []
                m = m + 1  # this will be placeholder for rows [happy, grumpy, hungry]
                m2 = 1  # this will be placeholder for rows [happy, grumpy, hungry]
                E = 0.0
                for s2 in self.emissions.keys():
                    if m2 == 1:
                        sub_emiss = self.emissions[s]
                        if O[i - 1] in sub_emiss :
                            E = sub_emiss[O[i - 1]]
                        else :
                            E = 0
                    sub_trans = self.transitions[s2]
                    if s in sub_trans:
                        T = sub_trans[s]
                    else:
                        T = 0
                    thing = O[i - 1]
                    prev = N[m2, i - 1]
                    prod = prev * float(T) * float(E)
                    prod_list.append(prod)
                    m2 = m2 + 1
                val = max(prod_list)
                N[m, i] = val
                bp[m, i] = prod_list.index(val) + 1

        predicted_list = []
        semi = N[:,(col-1)]
        bestN = max(semi)
        best = list(semi).index(bestN)
        j = len(O)
        while j > 0 :
            predicted_list.append(int(best))
            best = bp[int(best), j]
            j = j -1

        #populate list with state names
        predicted_states = []
        states = list(self.emissions.keys())
        for s in predicted_list :
            predicted_states.append(states[s-1])

        predicted_states.reverse()

        return predicted_states


if __name__ == "__main__" :
    filename = sys.argv[1]
    h = HMM()
    h.load(filename)

    if sys.argv[2] == '--generate' :
        l = h.generate(int(sys.argv[3]))
        print(l)
    elif sys.argv[2] == '--forward' :
        file = open(sys.argv[3])
        for line in file:
            words = []
            words = line.split()
            out = ''
            if len(words) > 0:
                for word in words:
                    out = out + word + ' '
                l = Sequence('', out)
                n = h.forward(l)
                print("Most probable state:")
                print(n)
                if filename == 'lander' :
                    if n == '4,3' or n == '3,4' or n == '2,5' or n == '4,4' or n == '5,5' :
                        print("It is safe to land!")
                    else:
                        print("Oh no! It is not safe to land!")
    elif sys.argv[2] == '--viterbi' :
        file = open(sys.argv[3])
        for line in file:
            words = []
            words = line.split()
            out = ''
            if len(words) > 0:
                for word in words:
                    out = out + word + ' '
                l = Sequence('', out)
                n = h.viterbi(l)
                print(n)


